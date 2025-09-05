# 🏪 Shop Owner Features Guide

## New Features Added

### 1. 📝 Daily Transaction Entry
Shop owners can now easily record daily sales and purchases with automatic stock updates.

**Features:**
- Record sales (reduces stock)
- Record purchases/restocking (increases stock)
- Automatic stock level updates
- Real-time low stock alerts

**How to Use:**
```bash
# Option 1: Command Line Interface
python main.py
# Select option 6: Record Daily Sales/Purchases

# Option 2: Web Interface (Easier)
streamlit run transaction_entry.py
```

### 2. ⚙️ Customizable Settings
All system settings are now editable by the shop owner without touching code.

**Configurable Settings:**
- Low stock threshold (default: 10 units)
- Critical stock threshold (default: 5 units)
- Reorder buffer days (default: 7 days)
- Email notification settings
- Daily order limits and budgets
- Auto-execution preferences

**How to Update Settings:**
- **Via Web Interface:** Use the Settings tab in `streamlit run transaction_entry.py`
- **Via Command Line:** Run `python main.py` and select option 7
- **Direct Edit:** Modify `config.json` file

### 3. 📧 Automated Email Alerts
Automatic email notifications when stock reaches threshold levels.

**Email Features:**
- Professional HTML email templates
- Customizable recipient lists
- Detailed product information
- Recommended actions
- Test email functionality

## 📋 Setup Instructions

### 1. Configure Email (Required for Alerts)

Update your `.env` file with your email credentials:

```properties
# Email settings for notifications
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_gmail_app_password
```

**For Gmail Users:**
1. Enable 2-factor authentication
2. Generate an "App Password" in Google Account settings
3. Use the app password, not your regular password

### 2. Update Recipients

Edit `config.json` to set who receives alerts:

```json
{
  "notification_settings": {
    "stock_alert_recipients": [
      "manager@yourstore.com",
      "owner@yourstore.com"
    ]
  }
}
```

### 3. Set Your Thresholds

Customize stock thresholds for your business:

```json
{
  "inventory_settings": {
    "default_low_stock_threshold": 10,
    "default_critical_stock_threshold": 5,
    "default_reorder_buffer_days": 7
  }
}
```

## 🚀 Quick Start Guide

### Daily Routine for Shop Owners:

1. **Start Your Day:**
   ```bash
   streamlit run transaction_entry.py
   ```

2. **Record Transactions:**
   - Go to "Enter Transactions" tab
   - Select Sale or Purchase
   - Choose product and enter quantity
   - Click "Add Transaction"

3. **Check Stock Status:**
   - Go to "Stock Status" tab
   - Review low stock items (highlighted in red)
   - Plan reorders for tomorrow

4. **Monitor Email Alerts:**
   - Automatically receive emails when stock is low
   - Follow recommended actions in emails

## 📊 Transaction Types

### Sales Transaction
- **Purpose:** Record customer purchases
- **Effect:** Reduces stock levels
- **Required:** Product ID, Quantity
- **Result:** Stock decreased, sales data updated

### Purchase/Restock Transaction  
- **Purpose:** Record inventory restocking
- **Effect:** Increases stock levels
- **Required:** Product ID, Quantity
- **Optional:** Unit Cost (updates product cost)
- **Result:** Stock increased, cost updated if provided

## 🔧 Advanced Configuration

### Email Template Customization

The email subject can be customized in `config.json`:

```json
{
  "notification_settings": {
    "email_template_subject": "🚨 Stock Alert: {product_name} is running low!"
  }
}
```

### Per-Product Thresholds

You can set individual thresholds for specific products by editing the `stock.csv` file's `reorder_point` column.

### Automatic Actions

Configure what the system does automatically:

```json
{
  "auto_execute_critical": true,    // Auto-order critical items
  "auto_execute_high": false,       // Manual approval for high priority
  "max_daily_orders": 20,           // Limit orders per day
  "max_daily_budget": 50000.0       // Daily spending limit
}
```

## 📈 Reports and Analytics

### Daily Summary
- View today's transactions
- Check stock levels
- Review alerts sent

### Weekly Analytics  
- Trend analysis
- Performance metrics
- Optimization recommendations

## 🆘 Troubleshooting

### Email Not Working?
1. Check EMAIL_USER and EMAIL_PASSWORD in `.env`
2. For Gmail, use App Password, not regular password
3. Test configuration in the web interface
4. Check if 2FA is enabled on your account

### Transactions Not Processing?
1. Ensure `data/stock.csv` exists
2. Check product IDs match exactly
3. Verify sufficient stock for sales
4. Check file permissions

### Settings Not Saving?
1. Ensure `config.json` has write permissions
2. Check JSON format is valid
3. Restart the application after changes

## 📞 Support

For technical support:
1. Check the `inventory_system.log` file for errors
2. Verify all files in `data/` folder exist
3. Ensure Python packages are installed: `pip install -r requirements.txt`

---

**Happy Inventory Managing! 🎉**
