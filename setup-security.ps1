# 🔒 Security Setup Script for Windows
# This script helps secure your environment and set up Git hooks

Write-Host "🔐 Agentic Inventory Management - Security Setup" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (Test-Path ".env") {
    Write-Host "✅ Found .env file" -ForegroundColor Green
    
    # Check if .env is tracked by Git
    $trackedEnv = git ls-files .env 2>$null
    if ($trackedEnv) {
        Write-Host "⚠️  WARNING: .env is tracked by Git!" -ForegroundColor Red
        Write-Host "This is a security risk. Removing from Git tracking..." -ForegroundColor Yellow
        git rm --cached .env
        Write-Host "✅ Removed .env from Git tracking" -ForegroundColor Green
        Write-Host "⚠️  Remember to commit this change!" -ForegroundColor Yellow
    } else {
        Write-Host "✅ .env is not tracked by Git (Good!)" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️  .env file not found" -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        $createEnv = Read-Host "Would you like to create .env from .env.example? (y/N)"
        if ($createEnv -eq 'y' -or $createEnv -eq 'Y') {
            Copy-Item ".env.example" ".env"
            Write-Host "✅ Created .env from template" -ForegroundColor Green
            Write-Host "⚠️  IMPORTANT: Edit .env and add your real credentials!" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Checking Git hooks setup..." -ForegroundColor Cyan

# Check if pre-commit hook exists
if (Test-Path ".git\hooks\pre-commit") {
    Write-Host "✅ Git pre-commit hook installed" -ForegroundColor Green
} else {
    Write-Host "❌ Git pre-commit hook not found" -ForegroundColor Red
    Write-Host "This should have been created automatically." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Testing .gitignore configuration..." -ForegroundColor Cyan

# Check if .env is in .gitignore
$gitignoreContent = Get-Content ".gitignore" -Raw
if ($gitignoreContent -match "\.env") {
    Write-Host "✅ .env is in .gitignore" -ForegroundColor Green
} else {
    Write-Host "⚠️  .env not found in .gitignore, adding it..." -ForegroundColor Yellow
    Add-Content ".gitignore" "`n# Environment variables`n.env"
    Write-Host "✅ Added .env to .gitignore" -ForegroundColor Green
}

Write-Host ""
Write-Host "Security Checklist:" -ForegroundColor Cyan
Write-Host "==================" -ForegroundColor Cyan
Write-Host "[ ] 1. Edit .env with your Gmail App Password" -ForegroundColor Yellow
Write-Host "[ ] 2. Never commit .env to Git" -ForegroundColor Yellow
Write-Host "[ ] 3. Use App Passwords, not regular passwords" -ForegroundColor Yellow
Write-Host "[ ] 4. Review SECURITY.md for detailed guidelines" -ForegroundColor Yellow
Write-Host "[ ] 5. Test the pre-commit hook: git add .env; git commit -m 'test'" -ForegroundColor Yellow

Write-Host ""
Write-Host "📚 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Generate a Gmail App Password:" -ForegroundColor White
Write-Host "   https://myaccount.google.com/apppasswords" -ForegroundColor Blue
Write-Host "2. Edit .env with your credentials" -ForegroundColor White
Write-Host "3. Read SECURITY.md for complete security guidelines" -ForegroundColor White
Write-Host "4. Run: python main.py" -ForegroundColor White

Write-Host ""
Write-Host "✅ Security setup complete!" -ForegroundColor Green
