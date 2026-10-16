@echo off
setlocal
cd /d "%~dp0.."
cargo run -p substancecraft
if errorlevel 1 pause