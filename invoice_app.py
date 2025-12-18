from dotenv import load_dotenv

load_dotenv()

import io
import json
import os
from datetime import datetime, timedelta
import secrets
import time
import bcrypt
import fitz
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    jsonify,
    send_file,
    session,
    url_for,
)
import pandas as pd
from werkzeug.utils import secure_filename

from PIL import Image
import google.generativeai as genai
import logging
from functools import wraps
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
import ssl
import mysql.connector
from mysql.connector import pooling
from datetime import timedelta



from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def now_ist():
    return now_ist()




# Load environment variables
# load_dotenv()

# Configure the Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Allowed file extensions
ALLOWED_EXTENSIONS = {"pdf", "jpeg", "jpg", "png"}

# API Keys loaded from environment variables
API_KEYS = [os.getenv("API_KEY_1"), os.getenv("API_KEY_2")]

# Usage limit per API key per day
USAGE_LIMIT = 10

# MySQL Configuration
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DB"),
    "auth_plugin": "mysql_native_password",
    "pool_name": "my_pool",
    "pool_size": 5,
    "autocommit": True,
    "buffered": True,
}

# Connection Pool
cnxpool = mysql.connector.pooling.MySQLConnectionPool(**MYSQL_CONFIG)


print("Sender:", os.getenv("EMAIL_ADDRESS"))
print("Password:", os.getenv("EMAIL_PASSWORD"))


def get_db_connection():
    """Returns a new MySQL connection from the pool."""
    cnx = cnxpool.get_connection()
    return cnx


def fix_existing_limits():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET account_limit = 50 WHERE account_limit IS NULL OR account_limit < 50"
    )
    conn.commit()
    cursor.close()
    conn.close()
    print("User limits updated to 50")


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS contact_queries (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL,
        contact VARCHAR(50),
        message TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) UNIQUE,
        email VARCHAR(255) UNIQUE,
        password VARCHAR(255),
        phone VARCHAR(20),                     -- new phone column
        invoices_extracted INT DEFAULT 10,
        passports_extracted INT DEFAULT 10,
        account_limit INT DEFAULT 50,
        manual_limit INT DEFAULT 50,
        api_limit INT DEFAULT 50,
        role VARCHAR(50) DEFAULT 'user'
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS extraction_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        timestamp DATETIME,
        image_name VARCHAR(255),
        pages_extracted INT,
        extraction_type VARCHAR(50),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS api_keys (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT,
        api_key VARCHAR(255) UNIQUE,
        created_at DATETIME,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS api_usage (
        id INT AUTO_INCREMENT PRIMARY KEY,
        api_key VARCHAR(255) UNIQUE,
        count INT DEFAULT 0,
        last_used DATE,
        usage_limit INT,
        FOREIGN KEY (api_key) REFERENCES api_keys(api_key) ON DELETE CASCADE
    )
    """
    )

    # ensure admin user exists
    cursor.execute(
        "SELECT * FROM users WHERE username = 'Niraj' AND email = 'connect.aiindia@gmail.com'"
    )
    admin_user = cursor.fetchone()
    if not admin_user:
        cursor.execute(
            """
            INSERT INTO users (username, email, password, role)
            VALUES (
                'Niraj',
                'connect.aiindia@gmail.com',
                '$2b$12$W9ObhLPinBHdnkX1XK6.U.Ub5mgx33abZtEY7xOVfHz5CMnM5tdWm',
                'admin'
            )
        """
        )
        conn.commit()

    cursor.close()
    conn.close()


# Re-initialize the database
init_db()


# Re-initialize the database with the updated schema


# Initialize the Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("FLASK_SECRET_KEY", default="fallback-secret-key")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def check_usage(api_key):
    current_date = now_ist().date()
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    sql = "SELECT * FROM api_usage WHERE api_key = %s"
    cursor.execute(sql, (api_key,))
    usage_entry = cursor.fetchone()
    if not usage_entry:
        sql_insert = (
            "INSERT INTO api_usage (api_key, count, last_used) VALUES (%s, %s, %s)"
        )
        cursor.execute(sql_insert, (api_key, 0, current_date))
        conn.commit()
        usage_entry = {"count": 0, "last_used": current_date, "usage_limit": None}
    else:
        if usage_entry["last_used"] < current_date:
            sql_update = (
                "UPDATE api_usage SET count = 0, last_used = %s WHERE api_key = %s"
            )
            cursor.execute(sql_update, (current_date, api_key))
            conn.commit()
            usage_entry["count"] = 0
            usage_entry["last_used"] = current_date
    usage_limit = (
        usage_entry["usage_limit"]
        if usage_entry["usage_limit"] is not None
        else USAGE_LIMIT
    )
    result = usage_entry["count"] < usage_limit
    cursor.close()
    conn.close()
    return result


def api_key_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            return jsonify({"error": "Unauthorized: API Key is required"}), 401
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT * FROM api_usage WHERE api_key = %s"
        cursor.execute(sql, (api_key,))
        key_record = cursor.fetchone()
        if not key_record:
            cursor.close()
            conn.close()
            return jsonify({"error": "Unauthorized: Invalid API Key"}), 401
        if key_record["count"] >= (
            key_record["usage_limit"]
            if key_record["usage_limit"] is not None
            else USAGE_LIMIT
        ):
            cursor.close()
            conn.close()
            return jsonify({"error": "Unauthorized: API Key usage limit exceeded"}), 403
        sql_update = "UPDATE api_usage SET count = count + 1 WHERE api_key = %s"
        cursor.execute(sql_update, (api_key,))
        conn.commit()
        cursor.close()
        conn.close()
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
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == ".pdf":
            images = pdf_to_images_pymupdf(file_path)
        else:
            images = [Image.open(file_path)]
        input_prompt = """
        {
            "invoice_number": "<Invoice Number>",
            "invoice_date": "<Invoice Date>",
            "due_date": "<Due Date>",
            "customer_name": "<Customer Name>",
            "customer_email": "<Customer Email>",
            "customer_phone": "<Customer Phone>",
            "customer_address": "<Customer Address>",
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
        for i, img in enumerate(images):
            response = model.generate_content([input_prompt, img])
            response_text = response.text.strip("```").strip()
            if response_text.lower().startswith("json"):
                response_text = response_text[4:].strip()
            if "}" in response_text:
                json_end_index = response_text.rfind("}") + 1
                response_text = response_text[:json_end_index]
            try:
                page_data = json.loads(response_text)
                all_pages_data[f"page_{i+1}"] = page_data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on page {i+1}: {e}")
                all_pages_data[f"page_{i+1}"] = {
                    "error": "Invalid JSON returned from Gemini",
                    "details": response_text,
                }
        return all_pages_data
    except Exception as e:
        print(f"Error during invoice extraction: {e}")
        return {"error": f"Error: {str(e)}"}


logging.basicConfig(level=logging.DEBUG)


def extract_passport_fields(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    try:
        if file_ext == ".pdf":
            images = pdf_to_images_pymupdf(file_path)
        else:
            images = [Image.open(file_path)]
        input_prompt = """
        You are an AI assistant trained to extract structured data from U.S. passport images. Analyze the uploaded U.S. passport image and provide all relevant details in the following JSON format, excluding any fields with null values or "NA" values:

        {
            "passport_number": "<Passport Number>",
            "passport_type": "<Passport Type>",
            "issue_date": "<Issue Date>",
            "expiration_date": "<Expiration Date>",
            "first_name": "<First Name>",
            "last_name": "<Last Name>",
            "middle_name": "<Middle Name>",
            "date_of_birth": "<Date of Birth>",
            "place_of_birth": "<Place of Birth>",
            "gender": "<Gender>",
            "nationality": "<Nationality>",
            "passport_issuing_country": "<Passport Issuing Country>",
            "passport_issuing_authority": "<Passport Issuing Authority>",
            "passport_issuing_country_code": "<Passport Issuing Country Code>",
            "passport_holder_photo": "<Passport Holder Photo URL>",
            "address": "<Address>",
            "city": "<City>",
            "state": "<State>",
            "zip_code": "<Zip Code>",
            "country": "<Country>",
            "height": "<Height>",
            "eye_color": "<Eye Color>",
            "hair_color": "<Hair Color>",
            "signature": "<Signature URL>",
            "passport_barcode_data": "<Passport Barcode Data>",
            "emergency_contact": {
                "name": "<Emergency Contact Name>",
                "phone": "<Emergency Contact Phone>",
                "address": "<Emergency Contact Address>"
            },
            "visa_status": "<Visa Status>",
            "visa_type": "<Visa Type>",
            "visa_number": "<Visa Number>",
            "visa_expiry_date": "<Visa Expiry Date>",
            "visa_issuing_country": "<Visa Issuing Country>",
            "visa_issuing_authority": "<Visa Issuing Authority>"
        }

        **Instructions:**
        - If any field has a value that is **null** or contains the string **"NA"**, it should **not be included** in the final JSON output.
        - Only include fields that contain valid, non-null, and non-"NA" data.
        - Do not include any extra text or explanation—only provide the structured JSON output.
        """
        all_pages_data = {}
        for i, img in enumerate(images):
            response = model.generate_content([input_prompt, img])
            response_text = response.text.strip("```").strip()
            if response_text.lower().startswith("json"):
                response_text = response_text[4:].strip()
            if "}" in response_text:
                json_end_index = response_text.rfind("}") + 1
                response_text = response_text[:json_end_index]
            try:
                page_data = json.loads(response_text)
                all_pages_data[f"page_{i+1}"] = page_data
            except json.JSONDecodeError as e:
                print(f"JSON parsing error on page {i+1}: {e}")
                all_pages_data[f"page_{i+1}"] = {
                    "error": "Invalid JSON returned from Gemini",
                    "details": response_text,
                }
        return all_pages_data
    except Exception as e:
        print(f"Error during passport extraction: {e}")
        return {"error": f"Error: {str(e)}"}


@app.route("/process-invoice", methods=["POST"])
@api_key_required
def process_invoice():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            upload_message = {"message": f"File uploaded successfully: {filename}"}
            extracted_data = extract_invoice_fields(filepath)
            if extracted_data:
                return jsonify(extracted_data), 200
            else:
                logging.error(f"Failed to extract invoice data from {filename}")
                return jsonify({"error": "Failed to extract invoice data"}), 500
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        logging.error(f"Error during upload or extraction: {e}")
        return jsonify({"error": "Internal Server Error"}), 500


@app.route("/usage-count", methods=["GET"])
@api_key_required
def get_usage_count():
    api_key = request.headers.get("X-API-Key")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    sql = "SELECT * FROM api_usage WHERE api_key = %s"
    cursor.execute(sql, (api_key,))
    usage_entry = cursor.fetchone()
    cursor.close()
    conn.close()
    if usage_entry:
        return jsonify(
            {
                "api_key": api_key,
                "usage_count": usage_entry["count"],
                "last_used": usage_entry["last_used"].strftime("%Y-%m-%d"),
            }
        )
    else:
        return jsonify({"error": "API key not found"}), 404


@app.route("/all-usage-counts", methods=["GET"])
@api_key_required
def get_all_usage_counts():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT * FROM api_usage"
        cursor.execute(sql)
        usage_entries = cursor.fetchall()
        usage_data = []
        for entry in usage_entries:
            usage_data.append(
                {
                    "api_key": entry["api_key"],
                    "usage_count": entry["count"],
                    "last_used": entry["last_used"].strftime("%Y-%m-%d"),
                }
            )
        cursor.close()
        conn.close()
        return jsonify(usage_data), 200
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500


DMIN_API_KEY = os.getenv("ADMIN_API_KEY")


@app.route("/admin/dashboard")
def admin_dashboard():
    if "user" not in session or session.get("role") != "admin":
        flash("Unauthorized access.")
        return redirect(url_for("login"))
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # fetch all users
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()

        # fetch all api_keys
        cursor.execute("SELECT * FROM api_keys")
        api_keys = cursor.fetchall()

        cursor.close()
        conn.close()

        # group api_keys by user_id
        api_keys_dict = {}
        for key in api_keys:
            uid = key["user_id"]
            api_keys_dict.setdefault(uid, []).append(key)

        # build user_activity list and include phone
        user_activity = []
        for u in users:
            u_id = u["id"]
            user_activity.append(
                {
                    "username": u["username"],
                    "email": u["email"],
                    "phone": u.get("phone", "—"),  # ← new phone field
                    "invoices_extracted": u.get("invoices_extracted", 0),
                    "account_limit": u.get("account_limit", 3),
                    "api_keys": api_keys_dict.get(u_id, []),
                    "api_keys": api_keys_dict.get(u_id, []),
                    "user_id": u_id,
                }
            )

        return render_template("admin_dashboard.html", user_activity=user_activity)

    except Exception as e:
        flash(f"Error loading dashboard: {e}")
        return render_template("admin_dashboard.html", user_activity=[])


@app.route("/admin/update-user-limit", methods=["POST"])
def update_user_limit():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        new_account_limit = data.get("new_account_limit")
        app.logger.info(f"Request data: {data}")
        app.logger.info(
            f"Received user_id: {user_id}, New account limit: {new_account_limit}"
        )
        if not user_id or new_account_limit is None:
            return jsonify({"error": "Missing required parameters"}), 400
        try:
            user_id = int(user_id)
            new_account_limit = int(new_account_limit)
            if new_account_limit < 0:
                return (
                    jsonify({"error": "Account limit must be a positive integer"}),
                    400,
                )
        except ValueError:
            return jsonify({"error": "Account limit must be a valid integer"}), 400
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT * FROM users WHERE id = %s"
        cursor.execute(sql, (user_id,))
        user = cursor.fetchone()
        if not user:
            cursor.close()
            conn.close()
            return jsonify({"error": "User not found"}), 404
        app.logger.info(f"User found: {user}")
        sql_update = "UPDATE users SET account_limit = %s WHERE id = %s"
        cursor.execute(sql_update, (new_account_limit, user_id))
        conn.commit()
        modified_count = cursor.rowcount
        cursor.close()
        conn.close()
        app.logger.info(f"Update result: {modified_count}")
        if modified_count == 0:
            return jsonify({"error": "No changes made"}), 404
        app.logger.info(f"User {user_id}'s account limit updated: {new_account_limit}")
        return (
            jsonify(
                {
                    "message": f"Account limit updated to {new_account_limit} for user {user_id}"
                }
            ),
            200,
        )
    except Exception as e:
        app.logger.error(f"Error updating user limit: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/some-a-endpoint", methods=["GET"])
def some_api_endpoint():
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return jsonify({"error": "API key is missing"}), 400
    print(f"Received API Key: {api_key}")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    sql = "SELECT * FROM api_usage WHERE api_key = %s"
    cursor.execute(sql, (api_key,))
    current_data = cursor.fetchone()
    print(f"Current Data: {current_data}")
    if not current_data:
        cursor.close()
        conn.close()
        return jsonify({"error": "API key not found"}), 404
    sql_update = "UPDATE api_usage SET count = count + 1 WHERE api_key = %s"
    cursor.execute(sql_update, (api_key,))
    conn.commit()
    cursor.execute(sql, (api_key,))
    updated_data = cursor.fetchone()
    print(f"Updated Data After Increment: {updated_data}")
    cursor.close()
    conn.close()
    return jsonify(
        {"message": "API key usage incremented successfully", "data": updated_data}
    )


@app.route("/", methods=["GET"])
def index():
    if "user" in session:
        return redirect(
            url_for("home_logged_in")
        )  # Redirect to proper logged-in homepage
    return render_template("home.html", logged_in=False)


# New Invoice Route
@app.route("/invoice")
def invoice():
    return render_template("invoice.html")  # After login, redirect to upload_image


@app.route("/passport")
def passport():
    return render_template("passport.html")


from email_validator import validate_email, EmailNotValidError
import phonenumbers
from phonenumbers import NumberParseException


def validate_user_email(email):
    try:
        v = validate_email(email)
        return v.email, None
    except EmailNotValidError as e:
        return None, str(e)


def validate_phone_number(raw_phone, default_region="IN"):
    try:
        pn = phonenumbers.parse(raw_phone, default_region)
        if not phonenumbers.is_valid_number(pn):
            return None, "Phone number is not valid"
        e164 = phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164)
        return e164, None
    except NumberParseException as e:
        return None, str(e)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email_in = request.form["email"].strip()
        phone_in = request.form.get("phone", "").strip()
        password = request.form["password"]

        # 1) Validate email
        email, email_err = validate_user_email(email_in)
        if email_err:
            flash(f"Invalid email address: {email_err}", "error")
            return redirect(url_for("signup"))

        # 2) Validate phone
        phone, phone_err = validate_phone_number(phone_in)
        if phone_err:
            flash(f"Invalid phone number: {phone_err}", "error")
            return redirect(url_for("signup"))

        # 3) Hash password
        hashed_password = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            sql = """
                INSERT INTO users (username, email, password, phone)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (username, email, hashed_password, phone))
            conn.commit()
            cursor.close()
            conn.close()
            flash("Sign up successful! Please log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Unexpected Error: {e}", "error")
            # optionally log e
    return render_template("appsignup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        login_input = request.form["username"]  # Can be username OR email
        password = request.form["password"]
        remember = request.form.get("remember")

        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)

            # Check if input is email or username
            sql = "SELECT * FROM users WHERE username=%s OR email=%s"
            cursor.execute(sql, (login_input, login_input))
            user = cursor.fetchone()

            cursor.close()
            conn.close()

            if user and bcrypt.checkpw(
                password.encode("utf-8"), user["password"].encode("utf-8")
            ):
                session.permanent = True if remember == "yes" else False
                app.permanent_session_lifetime = timedelta(days=7)

                session["user"] = user["username"]
                session["role"] = user.get("role", "user")

                if session["role"] == "admin":
                    return redirect(url_for("admin_dashboard"))

                return redirect(url_for("dashboard"))
            else:
                flash("Invalid username/email or password.")
        except Exception as e:
            flash(f"Error: {e}")

    return render_template("applogin.html")


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"]
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (email,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            if user:
                session["reset_email"] = email
                flash("Please enter your new password.")
                return redirect(url_for("reset_password"))
            else:
                flash("No account found with this email.")
        except Exception as e:
            flash(f"Error: {e}")
    return render_template("appforget.html")


@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    if "reset_email" not in session:
        flash("Invalid reset attempt.")
        return redirect(url_for("forgot_password"))
    if request.method == "POST":
        new_password = request.form["password"]
        try:
            hashed_password = bcrypt.hashpw(
                new_password.encode("utf-8"), bcrypt.gensalt()
            )
            conn = get_db_connection()
            cursor = conn.cursor()
            sql = "UPDATE users SET password = %s WHERE email = %s"
            cursor.execute(
                sql, (hashed_password.decode("utf-8"), session["reset_email"])
            )
            conn.commit()
            cursor.close()
            conn.close()
            session.pop("reset_email", None)
            flash("Password reset successfully.")
            return redirect(url_for("login"))
        except Exception as e:
            flash(f"Error: {e}")
    return render_template("appreset.html")


@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    if "user" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("login"))

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT * FROM users WHERE username = %s"
        cursor.execute(sql, (session["user"],))
        user = cursor.fetchone()
        if not user:
            flash("User not found.", "error")
            return redirect(url_for("login"))

        total_extracted = user.get("invoices_extracted", 0) + user.get(
            "passports_extracted", 0
        )
        print("USER DATA:", user)
        print("TOTAL EXTRACTED:", total_extracted)
        print("ACCOUNT LIMIT FROM DB:", user.get("account_limit"))
        print("FINAL LIMIT USED:", user.get("account_limit") or 50)
        extraction_limit = user.get("account_limit") or 50

        if total_extracted >= extraction_limit:
            flash(
                "You have reached your extraction limit. Contact admin for more.",
                "error",
            )

            return redirect(url_for("dashboard"))

        if request.method == "POST":
            if session.get("processing", False):
                flash(
                    "Please wait until your current extraction is complete.", "warning"
                )
                return redirect(url_for("dashboard"))

            doc_type = request.form.get("doc_type")
            if not doc_type or doc_type not in ["invoice", "passport"]:
                flash("Please select a valid document type.", "error")
                return redirect(url_for("dashboard"))

            file = request.files.get("file")
            if not file or not allowed_file(file.filename):
                flash("Invalid file type. Please upload a PDF or image.", "error")
                return redirect(url_for("dashboard"))

            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            session["processing"] = True

            logging.info(f"Processing {doc_type} file: {filename}")
            if doc_type == "invoice":
                extracted_data = extract_invoice_fields(file_path)
                update_field = "invoices_extracted"
            else:  # doc_type == 'passport'
                extracted_data = extract_passport_fields(file_path)
                update_field = "passports_extracted"

            if not extracted_data or "error" in extracted_data:
                logging.error(
                    f"Extraction failed for {filename}: {extracted_data.get('error', 'Unknown error')}"
                )
                flash(f"Failed to extract data from the {doc_type}.", "error")
                session["processing"] = False
                return redirect(url_for("dashboard"))

            page_count = len(extracted_data)
            sql_update = f"UPDATE users SET {update_field} = {update_field} + %s WHERE username = %s"
            cursor.execute(sql_update, (page_count, session["user"]))
            sql_insert = "INSERT INTO extraction_history (user_id, timestamp, image_name, pages_extracted, extraction_type) VALUES (%s, %s, %s, %s, %s)"
            cursor.execute(
                sql_insert, (user["id"], now_ist(), filename, page_count, doc_type)
            )
            conn.commit()

            session["processing"] = False
            json_data = json.dumps(extracted_data, indent=4)
            logging.info(f"Successfully extracted data from {filename}")
            return render_template("result.html", json_data=json_data)

        return redirect(url_for("dashboard"))

    except mysql.connector.Error as db_err:
        logging.error(f"Database error in upload_image: {db_err}", exc_info=True)
        flash(f"Database error: {db_err}", "error")
        return redirect(url_for("dashboard"))

    except Exception as e:
        logging.error(f"Unexpected error in upload_image: {e}", exc_info=True)
        flash(f"Error processing file: {e}", "error")
        return redirect(url_for("dashboard"))

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        if session.get("processing", False):
            session["processing"] = False


@app.route("/download_file", methods=["POST"])
def download_file():
    invoice_data = request.form.get("json_data")
    file_format = request.form.get("file_format")
    if not invoice_data:
        return "No data available for download", 400
    try:
        parsed_data = json.loads(invoice_data)
    except json.JSONDecodeError:
        return "Invalid JSON data", 400
    if file_format == "json":
        json_str = json.dumps(parsed_data, indent=4)
        buffer = io.BytesIO(json_str.encode("utf-8"))
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name="invoice_data.json",
            mimetype="application/json",
        )
    elif file_format == "excel":
        if isinstance(parsed_data, dict) and all(
            key.startswith("page_") for key in parsed_data.keys()
        ):
            data_list = []
            for key, value in parsed_data.items():
                if isinstance(value, dict):
                    value["page"] = key
                data_list.append(value)
            df = pd.DataFrame(data_list)
        else:
            df = pd.DataFrame([parsed_data])
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name="invoice_data.xlsx",
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        if request.is_json:
            return jsonify({"error": "Unauthorized access. Please log in."}), 401
        flash("Please login first.", "warning")
        return redirect(url_for("login"))

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        sql = "SELECT * FROM users WHERE username = %s"
        cursor.execute(sql, (session["user"],))
        user = cursor.fetchone()

        if not user:
            flash("User not found.", "error")
            if request.is_json:
                return jsonify({"error": "User not found."}), 404
            return redirect(url_for("login"))

        username = user.get("username", "N/A")
        email = user.get("email", "N/A")
        invoices_count = user.get("invoices_extracted", 0)
        passports_count = user.get("passports_extracted", 0)

        sql_history = "SELECT * FROM extraction_history WHERE user_id = %s ORDER BY timestamp DESC"
        cursor.execute(sql_history, (user["id"],))
        history = cursor.fetchall()

        sql_api_key = "SELECT api_key FROM api_keys WHERE user_id = %s"
        cursor.execute(sql_api_key, (user["id"],))
        user_api_key = cursor.fetchone()
        api_key = user_api_key["api_key"] if user_api_key else None

        formatted_history = [
            {
               # "timestamp_str": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                 "timestamp_str": entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                "image_name": entry.get("image_name", "Unknown"),
                "pages_extracted": entry.get("pages_extracted", 0),
                "extraction_type": entry.get("extraction_type", "N/A"),
            }
            for entry in history
        ]

        if request.is_json:
            return jsonify(
                {
                    "username": username,
                    "email": email,
                    "invoices_count": invoices_count,
                    "passports_count": passports_count,
                    "history": formatted_history,
                    "api_key": api_key,
                }
            )

        return render_template(
            "dashboard.html",
            username=username,
            email=email,
            invoices_count=invoices_count,
            passports_count=passports_count,
            history=formatted_history,
            api_key=api_key,
        )

    except mysql.connector.Error as db_err:
        logging.error(f"Database error: {db_err}", exc_info=True)
        flash(f"Database error: {db_err}", "error")
        if request.is_json:
            return jsonify({"error": str(db_err)}), 500
        return render_template(
            "dashboard.html",
            username="N/A",
            email="N/A",
            invoices_count=0,
            passports_count=0,
            history=[],
            api_key=None,
        )

    except Exception as e:
        logging.error(f"Unexpected error in dashboard: {e}", exc_info=True)
        flash(f"Error: {e}", "error")
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        return render_template(
            "dashboard.html",
            username="N/A",
            email="N/A",
            invoices_count=0,
            passports_count=0,
            history=[],
            api_key=None,
        )

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route("/view_history")
def view_history():
    if "user" not in session:
        flash("Please login first.")
        return redirect(url_for("login"))
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        sql = "SELECT id FROM users WHERE username = %s"
        cursor.execute(sql, (session["user"],))
        user = cursor.fetchone()
        if user:
            sql_history = "SELECT * FROM extraction_history WHERE user_id = %s ORDER BY timestamp DESC"
            cursor.execute(sql_history, (user["id"],))
            invoice_history = cursor.fetchall()
            formatted_history = []
            for entry in invoice_history:
                entry["timestamp_str"] = entry["timestamp"].strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                formatted_history.append(entry)
            cursor.close()
            conn.close()
            return render_template(
                "view_history.html", invoice_history=formatted_history
            )
        else:
            cursor.close()
            conn.close()
            flash("User not found.")
    except Exception as e:
        flash(f"Error: {e}")
    return render_template("view_history.html", invoice_history=[])


@app.route("/api/extract_invoice", methods=["POST"])
def api_extract_invoice():
    if "api_key" not in request.headers:
        return jsonify({"error": "Missing API key"}), 401
    api_key = request.headers["api_key"]
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    sql = "SELECT * FROM api_keys WHERE api_key = %s"
    cursor.execute(sql, (api_key,))
    api_key_record = cursor.fetchone()
    if not api_key_record:
        cursor.close()
        conn.close()
        return jsonify({"error": "Invalid API key"}), 403
    file = request.files.get("file")
    if not file or not allowed_file(file.filename):
        cursor.close()
        conn.close()
        return (
            jsonify(
                {"error": "Invalid file type. Please upload a valid PDF or image."}
            ),
            400,
        )
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)
    invoice_data = extract_invoice_fields(file_path)
    if invoice_data:
        sql_update = (
            "UPDATE users SET invoices_extracted = invoices_extracted + 1 WHERE id = %s"
        )
        cursor.execute(sql_update, (api_key_record["user_id"],))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"success": True, "invoice_data": invoice_data}), 200
    else:
        cursor.close()
        conn.close()
        return (
            jsonify({"success": False, "error": "Failed to extract invoice data"}),
            500,
        )


@app.route("/generate_api_key", methods=["POST"])
def generate_api_key():
    if "user" not in session:
        return jsonify({"error": "Unauthorized. Please log in first."}), 401
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    sql = "SELECT id FROM users WHERE username = %s"
    cursor.execute(sql, (session["user"],))
    user = cursor.fetchone()
    if not user:
        cursor.close()
        conn.close()
        return jsonify({"error": "User not found."}), 404
    sql_check = "SELECT * FROM api_keys WHERE user_id = %s"
    cursor.execute(sql_check, (user["id"],))
    existing_api_key = cursor.fetchone()
    if existing_api_key:
        cursor.close()
        conn.close()
        return jsonify({"error": "You have already generated an API key."}), 400
    api_key = secrets.token_hex(32)
    sql_insert = (
        "INSERT INTO api_keys (user_id, api_key, created_at) VALUES (%s, %s, %s)"
    )
    cursor.execute(sql_insert, (user["id"], api_key, now_ist())
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"api_key": api_key, "success": True})


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully!")
    return redirect(url_for("index"))


@app.route("/admin/set-account-limit", methods=["POST"])
def set_account_limit():
    try:
        data = request.get_json()
        new_manual_limit = data.get("manual_limit", 50)
        new_api_limit = data.get("api_limit", 50)
        if new_manual_limit < 0 or new_api_limit < 0:
            return jsonify({"error": "Limits must be positive integers"}), 400
        total_account_limit = new_manual_limit + new_api_limit
        conn = get_db_connection()
        cursor = conn.cursor()
        sql = "UPDATE users SET manual_limit = %s, api_limit = %s, account_limit = %s"
        cursor.execute(sql, (new_manual_limit, new_api_limit, total_account_limit))
        conn.commit()
        modified_count = cursor.rowcount
        cursor.close()
        conn.close()
        if modified_count == 0:
            return jsonify({"message": "No users updated"}), 404
        return (
            jsonify(
                {
                    "message": f"Account limits set to {total_account_limit} for all users"
                }
            ),
            200,
        )
    except Exception as e:
        app.logger.error(f"Error setting account limits: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/get_api_key", methods=["GET"])
def get_api_key():
    if "user" not in session:
        return jsonify({"error": "Unauthorized. Please log in first."}), 401
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    sql = "SELECT id FROM users WHERE username = %s"
    cursor.execute(sql, (session["user"],))
    user = cursor.fetchone()
    if not user:
        cursor.close()
        conn.close()
        return jsonify({"error": "User not found."}), 404
    sql_api = "SELECT api_key FROM api_keys WHERE user_id = %s"
    cursor.execute(sql_api, (user["id"],))
    api_key = cursor.fetchone()
    cursor.close()
    conn.close()
    if api_key:
        return jsonify({"api_key": api_key["api_key"], "success": True})
    else:
        return jsonify({"error": "No API key found."}), 404


@app.route("/api/routes", methods=["GET"])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append(
            {"endpoint": rule.endpoint, "methods": list(rule.methods), "url": str(rule)}
        )
    return jsonify({"routes": routes})


port = int(os.getenv("PORT", 8080))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/pricing")
def pricing():
    return render_template("pricing.html")


@app.route("/contact_sales")
def contact_sales():
    return render_template("about.html")


# Add this at the end of your application.py and run once


import os

# from dotenv import load_dotenv
# load_dotenv()
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Print current working directory to confirm
print("Current working directory:", os.getcwd())

# Load .env file
load_dotenv()  # Looks for .env in the current working directory

# Email config
# Email Configuration
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")  # e.g., 'your_email@gmail.com'
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # e.g., Gmail app-specific password
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")  # e.g., 'recipient@example.com'
SMTP_SERVER = "smtp.gmail.com"  # Adjust for your email provider
SMTP_PORT = 587


def send_email(email, contact, message):
    missing_configs = []
    if not EMAIL_ADDRESS:
        missing_configs.append("EMAIL_ADDRESS")
    if not EMAIL_PASSWORD:
        missing_configs.append("EMAIL_PASSWORD")
    if not RECIPIENT_EMAIL:
        missing_configs.append("RECIPIENT_EMAIL")
    if not SMTP_SERVER:
        missing_configs.append("SMTP_SERVER")
    if not SMTP_PORT:
        missing_configs.append("SMTP_PORT")

    if missing_configs:
        print(
            f"Email configuration missing: {', '.join(missing_configs)}. Check .env file."
        )
        return False

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = "New Query Submission from AI Infinite"

    # Format body nicely whether it's contact or demo
    if contact is not None:
        body = f"""
        Hello,

        A new query has been received through the AI Infinite contact form:

        Email: {email}
        Contact Number: {contact or 'Not provided'}
        Message: {message}

        Thank you,
        AI Infinite Team
        """
    else:
        body = message  # If it's demo, just use the already formatted message

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, int(SMTP_PORT))
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("Email sent successfully")
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False


@app.route("/submit", methods=["POST"])
def submit_query():
    print("Received form submission")
    email = request.form.get("email")
    contact = request.form.get("contact")
    message = request.form.get("message")
    print(f"Form data: email={email}, contact={contact}, message={message}")

    if not email or not message:
        print("Validation failed: Email or message missing")
        return (
            jsonify({"success": False, "message": "Email and message are required"}),
            400,
        )

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO contact_queries (email, contact, message) VALUES (%s, %s, %s)",
            (email, contact, message),
        )
        conn.commit()
        print("Query stored in database")
    except Exception as e:
        print(f"Database error: {e}")
        return jsonify({"success": False, "message": "Error storing query"}), 500
    finally:
        cursor.close()
        conn.close()

    if send_email(email, contact, message):
        print("Email sent successfully")
        return jsonify({"success": True, "message": "Query stored and email sent"})
    else:
        print("Email sending failed")
        return (
            jsonify(
                {"success": False, "message": "Query stored, but failed to send email"}
            ),
            200,
        )


@app.route("/schedule", methods=["POST"])
def schedule_demo():
    print("Received demo submission")
    name = request.form.get("name")
    email = request.form.get("company-email")
    timezone = request.form.get("timezone")

    if not email:
        print("Validation failed: Email missing")
        return jsonify({"success": False, "message": "Email is required"}), 400

    message = (
        "Hello,\n"
        "A new demo request has been issued by\n\n"
        f"Name: {name or 'Not provided'}\n"
        f"Email: {email}\n"
        f"Timezone: {timezone or 'Not specified'}\n\n"
        "Thank you."
    )

    if send_email(email, None, message):
        print("Demo email sent successfully")
        return jsonify(
            {"success": True, "message": "Demo request processed and email sent"}
        )
    else:
        print("Demo email sending failed")
        return (
            jsonify(
                {
                    "success": False,
                    "message": "Demo request processed, but failed to send email",
                }
            ),
            200,
        )


@app.route("/home")
def home_logged_in():
    if "user" not in session:
        print("User not in session. Redirecting to index.")
        return redirect(url_for("index"))
    print(f"User is in session: {session['user']}")
    return render_template("home.html", logged_in=True, username=session["user"])


@app.context_processor
def inject_user():
    return {"logged_in": "user" in session, "username": session.get("user")}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
    app.run(host="0.0.0.0", port=port, debug=True)
