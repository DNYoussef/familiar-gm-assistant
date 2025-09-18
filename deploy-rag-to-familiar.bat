@echo off
REM Deploy RAG System to Familiar Project
REM Research Princess Deployment Script

echo ===============================================
echo   RESEARCH PRINCESS RAG DEPLOYMENT SCRIPT
echo   Deploying Hybrid RAG System to Familiar
echo ===============================================

set SOURCE_DIR=%~dp0..\src\rag-system
set TARGET_DIR=C:\Users\17175\Desktop\familiar\src\rag

echo.
echo [1/6] Checking source directory...
if not exist "%SOURCE_DIR%" (
    echo ERROR: Source directory not found: %SOURCE_DIR%
    pause
    exit /b 1
)

echo [2/6] Checking target directory...
if not exist "C:\Users\17175\Desktop\familiar" (
    echo ERROR: Familiar project not found
    pause
    exit /b 1
)

echo [3/6] Creating RAG directory structure...
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
if not exist "%TARGET_DIR%\scripts" mkdir "%TARGET_DIR%\scripts"
if not exist "%TARGET_DIR%\tests" mkdir "%TARGET_DIR%\tests"
if not exist "%TARGET_DIR%\docs" mkdir "%TARGET_DIR%\docs"

echo [4/6] Copying RAG system files...
copy "%SOURCE_DIR%\*.js" "%TARGET_DIR%\" /Y
copy "%SOURCE_DIR%\package.json" "%TARGET_DIR%\" /Y
copy "%SOURCE_DIR%\.env.example" "%TARGET_DIR%\" /Y

echo [5/6] Copying documentation...
copy "%SOURCE_DIR%\deployment-guide.md" "%TARGET_DIR%\docs\" /Y

echo [6/6] Creating integration scripts...

REM Create package.json integration script
echo const fs = require('fs'); > "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo const path = require('path'); >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo. >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo // Add RAG dependencies to main package.json >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo const familiarPackage = JSON.parse(fs.readFileSync('../../package.json', 'utf8')); >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo const ragPackage = JSON.parse(fs.readFileSync('../package.json', 'utf8')); >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo. >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo familiarPackage.dependencies = { ...familiarPackage.dependencies, ...ragPackage.dependencies }; >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo fs.writeFileSync('../../package.json', JSON.stringify(familiarPackage, null, 2)); >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"
echo console.log('✅ RAG dependencies integrated into Familiar package.json'); >> "%TARGET_DIR%\scripts\integrate-with-familiar.js"

REM Create setup script
echo @echo off > "%TARGET_DIR%\scripts\setup-rag.bat"
echo echo Setting up RAG system for Familiar... >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo. >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo cd /d "%%~dp0.." >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo. >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo echo [1/4] Installing dependencies... >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo npm install >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo. >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo echo [2/4] Setting up environment... >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo if not exist .env copy .env.example .env >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo. >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo echo [3/4] Validating system... >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo npm run validate-system >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo. >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo echo [4/4] RAG system ready! >> "%TARGET_DIR%\scripts\setup-rag.bat"
echo echo Please edit .env with your API keys >> "%TARGET_DIR%\scripts\setup-rag.bat"

echo.
echo ===============================================
echo   DEPLOYMENT COMPLETE!
echo ===============================================
echo.
echo RAG System deployed to: %TARGET_DIR%
echo.
echo Next steps:
echo 1. cd "%TARGET_DIR%"
echo 2. Run setup: scripts\setup-rag.bat
echo 3. Configure .env with API keys
echo 4. Test: npm run validate-system
echo.
echo Research Princess Mission: ACCOMPLISHED ✅
echo Hybrid RAG system ready for Familiar integration!
echo.

pause