"""
Bangla Distill — Academic PDF Report Generator
Converts high-fidelity HTML/CSS reports with Bengali Unicode support to PDF via Chrome.
"""

import os
import sys
import subprocess
import pathlib

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

def find_browser():
    for p in CHROME_PATHS:
        if os.path.exists(p):
            return p
    raise RuntimeError("No compatible headless browser (Chrome or Edge) found for PDF generation.")

def html_to_pdf(html_content: str, output_pdf_path: str, save_html_copy: bool = True) -> str:
    browser_exe = find_browser()
    abs_pdf_path = os.path.abspath(output_pdf_path)
    os.makedirs(os.path.dirname(abs_pdf_path), exist_ok=True)
    
    html_path = abs_pdf_path.replace(".pdf", ".html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    html_uri = pathlib.Path(html_path).resolve().as_uri()
    
    cmd = [
        browser_exe,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={abs_pdf_path}",
        html_uri
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(abs_pdf_path) or os.path.getsize(abs_pdf_path) == 0:
        raise RuntimeError(f"Failed to generate PDF at {abs_pdf_path}. Browser error: {res.stderr}")
        
    if not save_html_copy and os.path.exists(html_path):
        try:
            os.remove(html_path)
        except Exception:
            pass
            
    print(f"✅ Generated PDF report: {abs_pdf_path} ({os.path.getsize(abs_pdf_path):,} bytes)")
    return abs_pdf_path

if __name__ == "__main__":
    test_html = "<html><body><h1>Bangla Distill PDF Engine Ready</h1></body></html>"
    out = html_to_pdf(test_html, "./test_output.pdf", save_html_copy=False)
    print("Test passed:", out)
    if os.path.exists("./test_output.pdf"):
        os.remove("./test_output.pdf")
