#!/bin/bash
# 🔒 Security Setup Script for Unix/Linux/Mac
# This script helps secure your environment and set up Git hooks

echo "🔐 Agentic Inventory Management - Security Setup"
echo "================================================="
echo ""

# Check if .env exists
if [ -f ".env" ]; then
    echo "✅ Found .env file"
    
    # Check if .env is tracked by Git
    if git ls-files --error-unmatch .env > /dev/null 2>&1; then
        echo "⚠️  WARNING: .env is tracked by Git!"
        echo "This is a security risk. Removing from Git tracking..."
        git rm --cached .env
        echo "✅ Removed .env from Git tracking"
        echo "⚠️  Remember to commit this change!"
    else
        echo "✅ .env is not tracked by Git (Good!)"
    fi
else
    echo "⚠️  .env file not found"
    if [ -f ".env.example" ]; then
        read -p "Would you like to create .env from .env.example? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp .env.example .env
            echo "✅ Created .env from template"
            echo "⚠️  IMPORTANT: Edit .env and add your real credentials!"
        fi
    fi
fi

echo ""
echo "Checking Git hooks setup..."

# Make pre-commit hook executable
if [ -f ".git/hooks/pre-commit" ]; then
    chmod +x .git/hooks/pre-commit
    echo "✅ Git pre-commit hook installed and made executable"
else
    echo "❌ Git pre-commit hook not found"
    echo "This should have been created automatically."
fi

echo ""
echo "Testing .gitignore configuration..."

# Check if .env is in .gitignore
if grep -q "^\.env$" .gitignore; then
    echo "✅ .env is in .gitignore"
else
    echo "⚠️  .env not found in .gitignore, adding it..."
    echo -e "\n# Environment variables\n.env" >> .gitignore
    echo "✅ Added .env to .gitignore"
fi

echo ""
echo "Security Checklist:"
echo "=================="
echo "[ ] 1. Edit .env with your Gmail App Password"
echo "[ ] 2. Never commit .env to Git"
echo "[ ] 3. Use App Passwords, not regular passwords"
echo "[ ] 4. Review SECURITY.md for detailed guidelines"
echo "[ ] 5. Test the pre-commit hook: git add .env && git commit -m 'test'"

echo ""
echo "📚 Next Steps:"
echo "1. Generate a Gmail App Password:"
echo "   https://myaccount.google.com/apppasswords"
echo "2. Edit .env with your credentials"
echo "3. Read SECURITY.md for complete security guidelines"
echo "4. Run: python main.py"

echo ""
echo "✅ Security setup complete!"
