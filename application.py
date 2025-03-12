import io
import json
import os
from datetime import datetime, timedelta
import secrets
import time
import bcrypt
import fitz
from flask import Flask, flash, redirect, render_template, request, jsonify, send_file, session, url_for
import pandas as pd
from werkzeug.utils import secure_filename
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import logging
from functools import wraps
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash


# Load environment variables
load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

ADMIN_API_KEY = os.getenv('ADMIN_API_KEY')


# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'jpeg', 'jpg', 'png'}

# API Keys loaded from environment variables
API_KEYS = [os.getenv('API_KEY_1'), os.getenv('API_KEY_2')]

# Usage limit per API key per day (example: 100 requests per day)
USAGE_LIMIT = 10

# Initialize MongoDB connection
client = MongoClient(os.getenv('MONGO_URI'))  # MongoDB URI from .env
db = client['gemini_db']  # Database name
usage_collection = db['api_usage']  # Collection to store API key usage data

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  

app.secret_key = os.getenv('FLASK_SECRET_KEY', default='fallback-secret-key')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extension checking function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API Key Authentication Function with usage limit check
def check_usage(api_key):
    current_date = datetime.now().date()  # Get current date
    usage_entry = usage_collection.find_one({'api_key': api_key})

    if not usage_entry:
        # Initialize the usage data if the API key doesn't exist in the collection
        usage_collection.insert_one({
            'api_key': api_key,
            'count': 0,
            'last_used': current_date.strftime('%Y-%m-%d')  # Convert to string
        })
        usage_entry = {
            "count": 0,
            "last_used": current_date.strftime('%Y-%m-%d')  # Convert to string
        }

    # Reset usage count if the day has changed
    if usage_entry["last_used"] != current_date.strftime('%Y-%m-%d'):  # Compare as strings
        # Only reset the count if the day has actually changed, do not reset on app restart
        usage_collection.update_one(
            {'api_key': api_key},
            {'$set': {'count': 0, 'last_used': current_date.strftime('%Y-%m-%d')}}  # Convert to string
        )
        usage_entry["count"] = 0
        usage_entry["last_used"] = current_date.strftime('%Y-%m-%d')  # Convert to string

    # Fetch the updated usage limit from the database (not hardcoded)
    usage_limit = usage_entry.get("usage_limit", USAGE_LIMIT)

    # Check if usage count has exceeded the limit
    if usage_entry["count"] >= usage_limit:
        return False  # Exceeded limit
    return True  # Below limit

def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "Unauthorized: API Key is required"}), 401

        # Check if the API key exists in the database
        key_record = db.api_usage.find_one({"api_key": api_key})
        if not key_record:
            return jsonify({"error": "Unauthorized: Invalid API Key"}), 401

        # Optional: Check if the key's usage limit has been reached
        if key_record['count'] >= key_record['usage_limit']:
            return jsonify({"error": "Unauthorized: API Key usage limit exceeded"}), 403
        usage_collection.update_one(
            {'api_key': api_key},
            {'$inc': {'count': 1}}
        )

        return f(*args, **kwargs)

    return decorated_function



def pdf_to_images_pymupdf(file_path):
    doc = fitz.open(file_path)
    images = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap()
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        images.append(image)
    return images
def extract_invoice_fields(file_path):
    """
    Extract invoice fields from an image or a PDF.
    Returns a dictionary with keys like "page_1", "page_2", etc.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        # Convert PDF pages to images or wrap single image in a list
        if file_ext == '.pdf':
            images = pdf_to_images_pymupdf(file_path)
        else:
            images = [Image.open(file_path)]
        
        # Input prompt for the AI model
        input_prompt = """
        
        {
            "invoice_number": "<Invoice Number>",
            "invoice_date": "<Invoice Date>",
            "due_date": "<Due Date>",
            "customer_name": "<Customer Name>",
            "customer_email": "<Customer Email>",
            "customer_phone": "<Customer Phone>",
            "customer_address": "<Customer Address>"
            "customer_state": "<Customer State>",
            "customer_gstin": "<Tax ID>",
            "billing_address": "<Billing Address>",
            "shipping_address": "<Shipping Address>",
            "seller_name": "<Seller Name>",
            "seller_email": "<Seller Email>",
            "seller_phone": "<Seller Phone>",
            "seller_address": "<Seller Address>",
            "seller_gstin": "<Tax ID>",
            "vat_number": "<VAT Number>",
            "purchase_order_number": "<Purchase Order Number>",
            "invoice_reference": "<Invoice Reference Number>",
            "items": [
                {
                    "description": "<Item Description>",
                    "quantity": <Item Quantity>,
                    "unit_price": <Item Unit Price>,
                    "total_price": <Item Total Price>,
                    "hsn_sac": "<HSN/SAC Code>",
                    "cgst_rate": <CGST Rate>,
                    "cgst_amount": <CGST Amount>,
                    "sgst_rate": <SGST Rate>,
                    "sgst_amount": <SGST Amount>,
                    "igst_rate": <IGST Rate>,
                    "igst_amount": <IGST Amount>,
                    "total_tax": <Total Tax (CGST + SGST + IGST)>,
                    "item_code": "<Item Code>",
                    "product_code": "<Product Code>",
                    "sku": "<SKU>"
                }
            ],
            "item_subtotal": <Subtotal Before Tax>,
            "tax_amount": <Tax Amount>,
            "tax_rate": <Tax Rate>,
            "discount_amount": <Discount Amount>,
            "total_amount": <Total Invoice Amount>,
            "payment_terms": "<Payment Terms>",
            "payment_method": "<Payment Method>",
            "bank_details": {
                "bank_name": "<Bank Name>",
                "account_number": "<Account Number>",
                "routing_number": "<Routing Number>",
                "iban": "<IBAN>",
                "swift_code": "<SWIFT/BIC Code>"
            },
            "payment_due_date": "<Payment Due Date>",
            "currency": "<Currency>",
            "exchange_rate": <Exchange Rate>,
            "shipping_cost": <Shipping Cost>,
            "handling_fee": <Handling Fee>,
            "freight_charges": <Freight Charges>,
            "customs_duties": <Customs Duties>,
            "insurance": <Insurance Charges>,
            "total_paid": <Amount Paid>,
            "balance_due": <Remaining Balance>,
            "notes": "<Additional Notes>",
            "footer": "<Footer Information>",
            "comments": "<Comments/Instructions>",
            "additional_information": "<Any other information>",
            "status": "<Invoice Status (e.g., Paid, Unpaid, Pending)>",
            "payment_reference": "<Payment Reference Number>"
        }
        
            **Instructions:**
        - If any field has a value that is **null** or contains the string **"NA"**, it should **not be included** in the final JSON output.
        - Only include fields that contain valid, non-null, and non-"NA" data.
        - Do not include any extra text or explanation—only provide the structured JSON output.
        
        """

        all_pages_data = {}
        # Process each image/page individually
        for i, img in enumerate(images):
            response = model.generate_content([input_prompt, img])
            response_text = response.text.strip('```').strip()
            # Remove any leading "json" text if present
            if response_text.lower().startswith('json'):
                response_text = response_text[4:].strip()
            # Ensure we cut the text off at the end of a JSON block
            if '}' in response_text:
                json_end_index = response_text.rfind('}') + 1
                response_text = response_text[:json_end_index]
            try:
                page_data = json.loads(response_text)
                all_pages_data[f"page_{i+1}"] = page_data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on page {i+1}: {e}")
                all_pages_data[f"page_{i+1}"] = {
                    'error': 'Invalid JSON returned from Gemini',
                    'details': response_text
                }
        return all_pages_data
    except Exception as e:
        print(f"Error during invoice extraction: {e}")
        return {'error': f'Error: {str(e)}'}



# Route to handle image upload and extraction request
logging.basicConfig(level=logging.DEBUG)

@app.route('/process-invoice', methods=['POST'])  # Updated route name
@api_key_required  # Ensure API key is required for access
def process_invoice():
    try:
        # Check if the request contains a file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        # If no file is selected, return an error
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Check if the file is allowed
        if file and allowed_file(file.filename):
            # Secure the filename and save the file 
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Send success message indicating file uploaded
            upload_message = {'message': f'File uploaded successfully: {filename}'}
            
            # Process the image using Gemini API to extract data
            extracted_data = extract_invoice_fields(filepath)
            
            if extracted_data:
                # Return the extracted JSON data after successful extraction
                return jsonify(extracted_data), 200
            else:
                # Log the error and send a failure response
                logging.error(f"Failed to extract invoice data from {filename}")
                return jsonify({'error': 'Failed to extract invoice data'}), 500
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        # Log the exception and return a generic error message
        logging.error(f"Error during upload or extraction: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500
    


@app.route('/usage-count', methods=['GET'])
@api_key_required
def get_usage_count():
    api_key = request.headers.get('X-API-Key')
    usage_entry = usage_collection.find_one({'api_key': api_key})

    if usage_entry:
        # Ensure 'last_used' is a datetime object, convert if it's a string

        
        last_used = usage_entry["last_used"]
        
        if isinstance(last_used, str):
            # If it's a string, convert it to a datetime object
            last_used = datetime.strptime(last_used, '%Y-%m-%d')

        # Now you can safely call strftime
        return jsonify({
            "api_key": api_key,
            "usage_count": usage_entry["count"],
            "last_used": last_used.strftime('%Y-%m-%d')  # Safely format the datetime
        })
    else:
        return jsonify({"error": "API key not found"}), 404




@app.route('/all-usage-counts', methods=['GET'])
@api_key_required  # Ensure the API key is valid
def get_all_usage_counts():
    try:
        usage_entries = usage_collection.find()  # Get all API key usage data
        usage_data = []
        
        for entry in usage_entries:
            usage_data.append({
                "api_key": entry["api_key"],
                "usage_count": entry["count"],
                "last_used": entry["last_used"]
            })
        
        return jsonify(usage_data), 200
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

DMIN_API_KEY = os.getenv('ADMIN_API_KEY')


# Enhanced Admin Dashboard Route
@app.route('/admin/dashboard', methods=['GET','POST'])
def admin_dashboard():
    if 'user' not in session or session.get('role') != 'admin':
        flash('Unauthorized access.')
        return redirect(url_for('login'))


    try:
        users_collection = db['users']
        api_keys_collection = db['api_keys']


        # Fetch users and convert _id to string
        users = list(users_collection.find({}, {'username': 1, 'email': 1, 'invoices_extracted': 1, 'account_limit': 1, '_id': 1}))
        for user in users:
            user['_id'] = str(user['_id'])  # Convert ObjectId to string


        # Fetch API keys and ensure user_id matches user collection
        api_keys = list(api_keys_collection.find({}, {'user_id': 1, 'api_key': 1, 'created_at': 1, 'usage_count': 1, 'usage_limit': 1, '_id': 0}))
        for key in api_keys:
            key['user_id'] = str(key['user_id'])  # Ensure user_id is a string
            if isinstance(key.get('created_at'), datetime):
                key['created_at'] = key['created_at'].strftime('%Y-%m-%d %H:%M:%S')


        # Create dictionary mapping users to their API keys
        api_keys_dict = {}
        for key in api_keys:
            user_id = key['user_id']
            if user_id not in api_keys_dict:
                api_keys_dict[user_id] = []
            api_keys_dict[user_id].append(key)


        # Prepare user activity data
        user_activity = []
        for user in users:
            user_activity.append({
                'username': user.get('username'),
                'email': user.get('email'),
                'invoices_extracted': user.get('invoices_extracted', 0),
                'account_limit': user.get('account_limit', 10),  # Default limit 50
                'api_keys': api_keys_dict.get(user['_id'], []),
                'user_id': user['_id']  # Include user ID for updating limits
            })


        return render_template('admin_dashboard.html', user_activity=user_activity)


    except Exception as e:
        flash(f"Error loading dashboard: {e}")
        return render_template('admin_dashboard.html', user_activity=[])
   
   
@app.route('/admin/update-user-limit', methods=['POST'])
def update_user_limit():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        new_account_limit = data.get('new_account_limit')  # Single limit


        # Log the received data
        app.logger.info(f"Request data: {data}")
        app.logger.info(f"Received user_id: {user_id}, New account limit: {new_account_limit}")


        if not user_id or new_account_limit is None:
            return jsonify({"error": "Missing required parameters"}), 400


        # Ensure new account limit is a valid positive integer
        try:
            new_account_limit = int(new_account_limit)


            if new_account_limit < 0:
                return jsonify({"error": "Account limit must be a positive integer"}), 400
        except ValueError:
            return jsonify({"error": "Account limit must be a valid integer"}), 400


        # Fetch user from database using user_id
        user = db['users'].find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({"error": "User not found"}), 404


        # Log the fetched user data
        app.logger.info(f"User found: {user}")


        # Update the user's account limit
        result = db['users'].update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {
                'account_limit': new_account_limit  # Store the new single limit
            }}
        )
       # Log the update result
        app.logger.info(f"Update result: {result.modified_count}")


        if result.modified_count == 0:
            return jsonify({"error": "No changes made"}), 404


        # Log the action
        app.logger.info(f"User {user_id}'s account limit updated: {new_account_limit}")


        # Confirm the update
        return jsonify({"message": f"Account limit updated to {new_account_limit} for user {user_id}"}), 200


    except Exception as e:
        app.logger.error(f"Error updating user limit: {e}")
        return jsonify({"error": str(e)}), 500




app.route('/some-api-endpoint', methods=['GET'])
def some_api_endpoint():
    api_key = request.headers.get('X-API-Key')
    if not api_key:
        return jsonify({"error": "API key is missing"}), 400

    # Log the API key
    print(f"Received API Key: {api_key}")

    # Check if the API key exists in the database
    current_data = db.api_usage.find_one({"api_key": api_key})
    print(f"Current Data: {current_data}")

    if not current_data:
        return jsonify({"error": "API key not found"}), 404

    # Increment the usage count (correct field name: 'count')
    update_result = db.api_usage.update_one(
        {"api_key": api_key},
        {"$inc": {"count": 1}}  # Correct field name is 'count', not 'usage_count'
    )
    
    # Check if the update was successful
    if update_result.modified_count > 0:
        updated_data = db.api_usage.find_one({"api_key": api_key})
        print(f"Updated Data After Increment: {updated_data}")
        return jsonify({"message": "API key usage incremented successfully", "data": updated_data})
    else:
        print("Update failed")
        return jsonify({"error": "Failed to update usage count"}), 500
    

    
@app.route('/')
def index():
    return render_template('applogin.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        try:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            users_collection = db['users']
            users_collection.insert_one({
                'username': username,
                'email': email,
                'password': hashed_password.decode('utf-8'),
                'invoices_extracted': 0,  # Initialize invoice count
                'extraction_history': []  # Initialize empty extraction history
            })
            flash('Sign up successful! Please log in.')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Unexpected Error: {e}")
    return render_template('appsignup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        try:
            users_collection = db['users']
            user = users_collection.find_one({'username': username})
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
                session['user'] = username
                session['role'] = user.get('role', 'user')  # Store role in session
                
                # Redirect admin to admin dashboard
                if session['role'] == 'admin':
                    return redirect(url_for('admin_dashboard'))
                
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password.')
        except Exception as e:
            flash(f"Error: {e}")
    
    return render_template('applogin.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        try:
            users_collection = db['users']
            user = users_collection.find_one({'email': email})
            if user:
                session['reset_email'] = email
                flash('Please enter your new password.')
                return redirect(url_for('reset_password'))
            else:
                flash('No account found with this email.')
        except Exception as e:
            flash(f"Error: {e}")
    return render_template('appforget.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if 'reset_email' not in session:
        flash('Invalid reset attempt.')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        new_password = request.form['password']
        try:
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            users_collection = db['users']
            users_collection.update_one(
                {'email': session['reset_email']},
                {'$set': {'password': hashed_password.decode('utf-8')}}
            )
            session.pop('reset_email', None)
            flash('Password reset successfully.')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Error: {e}")
    return render_template('appreset.html')

@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    # Ensure user is logged in
    if 'user' not in session:
        flash('Please login first.')
        return redirect(url_for('login'))

    users_collection = db['users']
    user = users_collection.find_one({'username': session['user']})
    if not user:
        flash('User not found.')
        return redirect(url_for('login'))

    # Get user's extraction count and limit
    invoices_extracted = user.get('invoices_extracted', 0)
    extraction_limit = user.get('account_limit', 3)  # Default extraction limit

    if invoices_extracted >= extraction_limit:
        flash('You have reached your invoice extraction limit. Contact admin for more.')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        if session.get('processing', False):
            flash('Please wait until your current extraction is complete.')
            return redirect(url_for('upload_image'))

        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            flash('Invalid file type. Please upload a PDF or image.')
            return redirect(url_for('upload_image'))

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        session['processing'] = True
        time.sleep(5)  # Simulate processing delay
        session['processing'] = False

        # Extract invoice data from the file
        invoice_data = extract_invoice_fields(file_path)
        if invoice_data:
            # Count the number of pages processed
            page_count = len(invoice_data)

            # Increment extraction count by the number of pages extracted and log history
            users_collection.update_one(
                {'username': session['user']},
                {'$inc': {'invoices_extracted': page_count},
                 '$push': {'extraction_history': {
                     'timestamp': datetime.now(),
                     'image_name': filename,
                     'pages_extracted': page_count
                 }}}
            )
            return render_template('result.html', json_data=json.dumps(invoice_data, indent=4))
        else:
            flash('Failed to extract data from the image.')

    return render_template('dashboard.html')

@app.route('/download_file', methods=['POST'])
def download_file():
    invoice_data = request.form.get('json_data')
    file_format = request.form.get('file_format')

    if not invoice_data:
        return "No data available for download", 400

    try:
        parsed_data = json.loads(invoice_data)
    except json.JSONDecodeError:
        return "Invalid JSON data", 400

    if file_format == 'json':
        # Convert the JSON data to a string and then encode it into bytes.
        json_str = json.dumps(parsed_data, indent=4)
        buffer = io.BytesIO(json_str.encode('utf-8'))
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name='invoice_data.json',  # For Flask >= 2.0. Use attachment_filename if Flask < 2.0.
            mimetype='application/json'
        )

    elif file_format == 'excel':
        # If the parsed data represents multi-page data (with keys like "page_1", "page_2", etc.)
        if isinstance(parsed_data, dict) and all(key.startswith("page_") for key in parsed_data.keys()):
            data_list = []
            for key, value in parsed_data.items():
                if isinstance(value, dict):
                    # Optionally add a 'page' column to indicate the page number.
                    value['page'] = key
                data_list.append(value)
            df = pd.DataFrame(data_list)
        else:
            # Single-page extraction
            df = pd.DataFrame([parsed_data])

        # Write the DataFrame to an Excel file in memory.
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name='invoice_data.xlsx',  # For Flask >= 2.0. Use attachment_filename if Flask < 2.0.
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    return "Unsupported file format", 400


@app.route('/dashboard', methods=['GET','POST'])
def dashboard():
    if 'user' not in session:
        if request.is_json:
            return jsonify({'error': 'Unauthorized access. Please log in.'}), 401
        flash('Please login first.')
        return redirect(url_for('login'))

    try:
        # Fetch user data from MongoDB
        users_collection = db['users']
        user = users_collection.find_one({'username': session['user']})

        if user:
            # Extract user details
            username = user.get('username', 'N/A')
            email = user.get('email', 'N/A')
            invoices_count = user.get('invoices_extracted', 0)  # Default to 0 if not set
            invoice_history = user.get('extraction_history', [])

            # Fetch the user's API key if it exists
            api_key = None
            api_keys_collection = db['api_keys']
            user_api_key = api_keys_collection.find_one({'user_id': user['_id']})
            if user_api_key:
                api_key = user_api_key.get('api_key')

            # Format the datetime objects into a human-readable format
            formatted_history = []
            for entry in invoice_history:
                if isinstance(entry, dict) and 'timestamp' in entry:
                    entry['timestamp_str'] = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    formatted_history.append(entry)

            # If the request is an API call (JSON requested), return data as JSON
            if request.is_json:
                return jsonify({
                    'username': username,
                    'email': email,
                    'invoices_count': invoices_count,
                    'invoice_history': formatted_history,
                    'api_key': api_key  # Return the API key if it exists
                })

            # Otherwise, render the HTML dashboard
            return render_template(
                'dashboard.html',
                username=username,
                email=email,
                invoices_count=invoices_count,
                invoice_history=formatted_history,
                api_key=api_key  # Pass API key to the template if it exists
            )
        else:
            if request.is_json:
                return jsonify({'error': 'User not found.'}), 404
            flash('User not found.')
            return redirect(url_for('login'))

    except Exception as e:
        # Handle errors and show a flash message
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        flash(f"Error: {e}")
        return render_template('dashboard.html', username='N/A', email='N/A', invoices_count=0, invoice_history=[])


    except Exception as e:
        # Handle errors and show a flash message
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        flash(f"Error: {e}")
        return render_template('dashboard.html', username='N/A', email='N/A', invoices_count=0, invoice_history=[])
    

@app.route('/view_history')
def view_history():
    if 'user' not in session:
        flash('Please login first.')
        return redirect(url_for('login'))

    try:
        # Fetch user data from the database
        users_collection = db['users']
        user = users_collection.find_one({'username': session['user']})

        if user:
            invoice_history = user.get('extraction_history', [])

            # Format the datetime objects into a human-readable format
            formatted_history = []
            for entry in invoice_history:
                if isinstance(entry, dict):
                    entry['timestamp_str'] = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    formatted_history.append(entry)

            return render_template(
                'view_history.html', 
                invoice_history=formatted_history  # Pass the formatted history to the template
            )
        else:
            flash('User  not found.')

    except Exception as e:
        flash(f"Error: {e}")

    return render_template('view_history.html', invoice_history=[])




@app.route('/api/extract_invoice', methods=['POST'])
def api_extract_invoice():
    """
    API endpoint for extracting invoice data.
    """
    if 'api_key' not in request.headers:
        return jsonify({'error': 'Missing API key'}), 401

    api_key = request.headers['api_key']

    # Verify API key
    api_keys_collection = db['api_keys']
    api_key_record = api_keys_collection.find_one({'api_key': api_key})
    if not api_key_record:
        return jsonify({'error': 'Invalid API key'}), 403

    # Validate and process uploaded file
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a valid PDF or image.'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract invoice data
    invoice_data = extract_invoice_fields(file_path)
    if invoice_data:
        # Increment user's invoice usage
        users_collection = db['users']
        users_collection.update_one(
            {'_id': api_key_record['user_id']},
            {'$inc': {'invoices_extracted': 1}}
        )

        return jsonify({'success': True, 'invoice_data': invoice_data}), 200
    else:
        return jsonify({'success': False, 'error': 'Failed to extract invoice data'}), 500


@app.route('/generate_api_key', methods=['POST'])
def generate_api_key():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized. Please log in first.'}), 401

    users_collection = db['users']
    user = users_collection.find_one({'username': session['user']})
    if not user:
        return jsonify({'error': 'User not found.'}), 404

    api_keys_collection = db['api_keys']
    existing_api_key = api_keys_collection.find_one({'user_id': user['_id']})
    if existing_api_key:
        return jsonify({'error': 'You have already generated an API key.'}), 400

    api_key = secrets.token_hex(32)
    api_keys_collection.insert_one({
        'user_id': user['_id'],
        'api_key': api_key,
        'created_at': datetime.utcnow()
    })

    return jsonify({"api_key": api_key, "success": True})

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Logged out successfully!')
    return redirect(url_for('login'))


@app.route('/admin/set-account-limit', methods=['POST'])
def set_account_limit():
    try:
        data = request.get_json()
        new_manual_limit = data.get('manual_limit', 50)  # Default to 50 if not provided
        new_api_limit = data.get('api_limit', 50)        # Default to 50 if not provided

        # Ensure new limits are valid positive integers
        if new_manual_limit < 0 or new_api_limit < 0:
            return jsonify({"error": "Limits must be positive integers"}), 400

        # Calculate total account limit
        total_account_limit = new_manual_limit + new_api_limit

        # Update account limits for all users
        result = db['users'].update_many(
            {},
            {'$set': {
                'manual_limit': new_manual_limit,
                'api_limit': new_api_limit,
                'account_limit': total_account_limit  # Update the account limit
            }}
        )

        if result.modified_count == 0:
            return jsonify({"message": "No users updated"}), 404

        return jsonify({"message": f"Account limits set to {total_account_limit} for all users"}), 200

    except Exception as e:
        app.logger.error(f"Error setting account limits: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/get_api_key', methods=['GET'])
def get_api_key():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized. Please log in first.'}), 401


    users_collection = db['users']
    user = users_collection.find_one({'username': session['user']})
    if not user:
        return jsonify({'error': 'User not found.'}), 404


    api_keys_collection = db['api_keys']
    api_key = api_keys_collection.find_one({'user_id': user['_id']})


    if api_key:
        return jsonify({'api_key': api_key['api_key'], 'success': True})
    else:
        return jsonify({'error': 'No API key found.'}), 404
        

@app.route('/api/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "url": str(rule)
        })
    return jsonify({"routes": routes})
    
port = int(os.getenv("PORT", 8080))





if __name__ == '__main__':
     port = int(os.environ.get("PORT", 8080))
     app.run(host="0.0.0.0", port=port)
