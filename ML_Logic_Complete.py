import json
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from googlesearch import search
from bs4 import BeautifulSoup
import io
import docx2txt
import PyPDF2
import requests
import time
import random
from datetime import datetime, timedelta

# --- NEW IMPORTS FOR ML/NLP ---
import spacy
from spacy.matcher import PhraseMatcher
# Load the small English model once on startup
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("WARNING: spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None
# ------------------------------

# --- FASTAPI SETUP ---
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GO SERVER CONFIG ---
GO_SERVER_URL = "http://localhost:8080/jobs-data"

# --- PYDANTIC MODELS ---
class Job(BaseModel):
    title: str
    company: str
    website: str
    skills: List[str]

# --- ML LOGIC (Implementation using spaCy) ---
def extract_keywords_from_resume(resume_text: str) -> List[str]:
    """
    Extracts technical skills and entities from the resume using spaCy's NLP capabilities.
    """
    if nlp is None:
        print("ERROR: spaCy model not loaded. Returning fallback keywords.")
        return ["Fallback", "NLP Error"]

    print("⚙️  Parsing resume to extract keywords using spaCy...")
    doc = nlp(resume_text)
    
    # 1. Use pre-trained NER for general entities (Name, Org, GPE)
    # This helps get general job/location keywords
    general_keywords = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "GPE", "PRODUCT", "JOB")]
    
    # 2. Use a simple, predefined list for common technical skills 
    # (Since spaCy's default model isn't trained specifically on tech skills)
    tech_skills = [
        "Python", "JavaScript", "React", "Angular", "Vue", "Node.js", 
        "SQL", "MongoDB", "AWS", "Azure", "Docker", "Kubernetes",
        "Machine Learning", "Data Science", "API", "FastAPI"
    ]
    
    # Simple deduplication and filtering based on presence in the text
    found_skills = set()
    for skill in tech_skills:
        if skill.lower() in resume_text.lower():
            found_skills.add(skill)
            
    # Combine lists and clean up
    all_keywords = list(set(general_keywords + list(found_skills)))
    
    # Final filter: remove keywords that are too short (like single letters or numbers)
    keywords = [kw for kw in all_keywords if len(kw.split()) > 1 or len(kw) > 3]

    print(f"✅ Extracted Keywords: {keywords}")
    return keywords

# --- ADVANCED WEB SCRAPING LOGIC (Unchanged from last step) ---
def is_recent_job_posting(text: str) -> bool:
    """
    Conceptual function to check if a job posting is recent based on text.
    """
    text_lower = text.lower()
    today = datetime.now()

    if "just posted" in text_lower or "new" in text_lower or "24 hours" in text_lower:
        return True

    for i in range(1, 8):
        if f"{i} day" in text_lower or f"{i}d" in text_lower:
            return True

    return False


def scrape_jobs(keywords: List[str]) -> Dict[str, List[str]]:
    """
    Automates a browser to scrape job listings with improved resilience.
    """
    print("Starting advanced job scraping process with Selenium...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        print(f"❌ Failed to initialize the browser driver: {e}")
        return {}

    results_by_keyword = {}

    for keyword in keywords:
        print(f"🔍 Searching for: {keyword}")
        results_by_keyword[keyword] = []
        try:
            for url in search(f"{keyword} job", num_results=5, lang="en"):
                print(f"  → {url}")
                try:
                    time.sleep(random.uniform(2, 5))
                    driver.get(url)
                    
                    soup = BeautifulSoup(driver.page_source, "html.parser")
                    text = soup.get_text(" ", strip=True).lower()

                    if any(word in text for word in ["apply now", "vacancy", "hiring", "job", "careers"]) and is_recent_job_posting(text):
                        print("    ✅ Job posting detected and is recent.")
                        results_by_keyword[keyword].append(url)
                    else:
                        print("    ℹ️ No recent job posting keywords found.")
                        
                except (TimeoutException, WebDriverException) as e:
                    print(f"    ⚠️ A browser error occurred while fetching {url}: {e}")
                    continue
                except Exception as e:
                    print(f"    ⚠️ An unexpected error occurred: {e}")
                    continue
        except Exception as e:
            print(f"❌ Search failed for '{keyword}': {e}")
            continue

    driver.quit()
    return results_by_keyword

# --- ROUTES ---
@app.post("/recommend-jobs")
async def recommend_jobs(file: UploadFile = File(...)):
    """
    Endpoint to receive resume file, extract skills, scrape jobs,
    and then send results to a Go server.
    """
    # 1. Read the file content
    file_content = await file.read()
    file_type = file.filename.split('.')[-1].lower()
    resume_text = ""
    if file_type == "pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            for page in pdf_reader.pages:
                resume_text += page.extract_text()
        except:
            return {"status": "error", "message": "Failed to parse PDF content."}
    elif file_type == "docx":
        try:
            # docx2txt.process expects a path or a BytesIO object for the docx file
            resume_text = docx2txt.process(io.BytesIO(file_content)).decode("utf-8")
        except:
            return {"status": "error", "message": "Failed to parse DOCX content."}
    else:
        # Fallback for plain text
        resume_text = file_content.decode("utf-8")

    # 2. Extract keywords from the resume
    keywords = extract_keywords_from_resume(resume_text)
    
    # 3. Scrape jobs from the internet based on the extracted keywords
    fetched_jobs = scrape_jobs(keywords)

    # 4. Send the scraped jobs as a JSON payload to the Go server
    try:
        print(f"📦 Sending results to Go server at {GO_SERVER_URL}")
        response = requests.post(GO_SERVER_URL, json=fetched_jobs, timeout=30)
        
        if response.status_code == 200:
            print("✅ Successfully sent job data to Go server.")
            return {"status": "success", "message": "Job data sent to Go server."}
        else:
            print(f"❌ Failed to send data. Status code: {response.status_code}")
            return {"status": "error", "message": "Failed to send data to Go server."}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred while connecting to the Go server: {e}")
        return {"status": "error", "message": f"Connection error: {e}"}