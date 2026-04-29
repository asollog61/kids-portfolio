@echo off
cd /d "%~dp0"
echo Pushing Kids Portfolio updates to GitHub...
git add -A
git commit -m "Update %date% %time%"
git push
echo.
echo Done! Streamlit Cloud will auto-redeploy in ~1 minute.
pause
