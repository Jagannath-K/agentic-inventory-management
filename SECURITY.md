# 🔒 Security Guidelines

## Email Configuration Security

This project uses email notifications for inventory alerts. To protect your credentials:

### ⚠️ IMPORTANT: Never Commit `.env` File

The `.env` file contains sensitive credentials and should **NEVER** be committed to Git.

### ✅ Current Security Status

- ✅ `.env` is listed in `.gitignore`
- ✅ `.env.example` provided as template (no real credentials)
- ✅ Code uses environment variables via `python-dotenv`

### 🔧 Setup Instructions

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Configure your Gmail App Password:**
   - Go to [Google Account Security](https://myaccount.google.com/security)
   - Enable 2-Step Verification (if not already enabled)
   - Go to [App Passwords](https://myaccount.google.com/apppasswords)
   - Generate a new App Password for "Mail"
   - Copy the 16-character password (spaces will be removed)

3. **Update `.env` with your credentials:**
   ```properties
   EMAIL_USER=your-email@gmail.com
   EMAIL_PASSWORD=your-16-char-app-password
   CRITICAL_ALERT_RECIPIENT=your-alert-email@gmail.com
   ```

4. **Verify `.env` is not tracked:**
   ```bash
   git status  # .env should NOT appear here
   git ls-files .env  # Should return nothing
   ```

### 🚨 If You Accidentally Committed Credentials

If you've already pushed credentials to GitHub:

1. **Revoke the compromised credentials immediately:**
   - Delete the Gmail App Password at [App Passwords](https://myaccount.google.com/apppasswords)
   - Generate a new one

2. **Remove from Git history:**
   ```bash
   # WARNING: This rewrites history!
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   
   # Force push (coordinate with team if applicable)
   git push origin --force --all
   ```

3. **Alternative: Use BFG Repo-Cleaner (easier):**
   ```bash
   # Install: https://rtyley.github.io/bfg-repo-cleaner/
   bfg --delete-files .env
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   git push origin --force --all
   ```

### 🔐 Best Practices

1. **Use App Passwords, not regular passwords** - More secure and can be revoked individually
2. **Never hardcode credentials** - Always use environment variables
3. **Use different passwords** - Don't reuse your email password
4. **Regular rotation** - Change App Passwords periodically
5. **Limit permissions** - Only give necessary access
6. **Monitor access** - Check [Recent Security Activity](https://myaccount.google.com/notifications)

### 📋 Pre-Commit Checklist

Before every commit:
- [ ] Verify `.env` is not staged: `git status`
- [ ] No credentials in code files
- [ ] `.env.example` has no real credentials
- [ ] Documentation doesn't contain secrets

### 🛡️ Additional Security Measures

Consider implementing:
- **Git hooks** to prevent `.env` commits
- **Secret scanning** tools (e.g., git-secrets, truffleHog)
- **Environment-specific configs** for dev/staging/prod
- **Vault solutions** for production (e.g., HashiCorp Vault, AWS Secrets Manager)

### 📞 Emergency Contact

If credentials are exposed:
1. Revoke immediately
2. Change all related passwords
3. Monitor for suspicious activity
4. Consider enabling additional security measures

## License

This security guide is part of the Agentic Inventory Management System project.
