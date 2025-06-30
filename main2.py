import os
import time
import re
import json
import urllib.parse
import http.client
from typing import List, Dict, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from serpapi import GoogleSearch
import google.generativeai as genai
from huggingface_hub import InferenceClient

# --- Configuration ---
load_dotenv()

# Environment variables with fallbacks
HF_TOKEN = os.getenv("HF_TOKEN") 
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # This seems like a Google Generative AI key, not for SerpApi
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY") 
RAPIDAPI_HOST = "fresh-linkedin-profile-data.p.rapidapi.com"

# Cache settings
CACHE_DIR = "cache"
CACHE_EXPIRATION_SECONDS = 24 * 60 * 60  # 24 hours

os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize Hugging Face Inference Client
try:
    hf_client = InferenceClient(
        provider="novita",
        api_key=HF_TOKEN,
    )
    HF_MODEL = "deepseek-ai/DeepSeek-V3-0324"
except Exception as e:
    print(f"Error initializing Hugging Face client: {e}")
    hf_client = None # Set to None if initialization fails

# --- LLM Helper ---

def get_llm_response(prompt: str) -> str:
    """Sends a user prompt to the configured Hugging Face model and returns the raw text."""
    if not hf_client:
        return "LLM service not available."
    try:
        res = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ LLM inference error: {e}")
        return ""

# --- Utilities ---

def extract_name_from_linkedin_url(url: str) -> str:
    """Derives a readable name from a LinkedIn profile URL slug as a fallback."""
    if not url:
        return "N/A"
    match = re.search(r"linkedin\.com/in/([^/?#]+)", url)
    if not match:
        return "N/A"
    slug = match.group(1)
    # Remove numbers and split by hyphen, then capitalize each part
    parts = [p for p in slug.split('-') if not re.fullmatch(r"\d+", p)]
    return " ".join(p.capitalize() for p in parts) or "N/A"

def parse_json_from_llm_response(text: str) -> Dict[str, Any]:
    """Extracts and parses a JSON object from an LLM's text response."""
    try:
        # LLMs often embed JSON within text, so we look for the first and last curly brace
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}. Raw text: {text[:200]}...") # Log partial text for debugging
    return {}

# --- Caching ---

def get_cache_path(key: str) -> str:
    """Generates a file path for a given cache key."""
    # Ensure keys are safe for filenames (e.g., replace / with _)
    safe_key = re.sub(r'[^\w\-_\.]', '_', key)
    return os.path.join(CACHE_DIR, f"{safe_key}.json")

def read_from_cache(key: str) -> Dict | None:
    """Reads data from cache if it exists and is not expired."""
    cache_path = get_cache_path(key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                if time.time() - data.get("timestamp", 0) < CACHE_EXPIRATION_SECONDS:
                    return data.get("content")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"❌ Error reading or parsing cache for {key}: {e}")
            # Optionally remove corrupted cache file
            os.remove(cache_path)
    return None

def write_to_cache(key: str, content: Dict):
    """Writes data to the cache with a timestamp."""
    try:
        with open(get_cache_path(key), 'w') as f:
            json.dump({"timestamp": time.time(), "content": content}, f, indent=2)
    except IOError as e:
        print(f"❌ Error writing to cache for {key}: {e}")

# --- Job Analysis ---

def analyze_job_description(job_desc: str) -> Dict[str, Any]:
    """Extracts key details from a job description using the LLM."""
    prompt = f"""
    You are an expert job analyst. Extract key details from the following job description.
    Return the information as a strict JSON object with the following keys:
    'title': (string) The job title.
    'company': (string) The hiring company's name.
    'location': (string) The primary work location.
    'search_keywords': (list of strings) 3-4 highly relevant keywords for searching candidates (e.g., "Machine Learning Engineer", "Python", "Deep Learning").
    'summary': (string) A concise summary of the role's responsibilities and requirements.

    Job Description:
    {job_desc}
    """
    print("🤖 Analyzing job description...")
    llm_response = get_llm_response(prompt)
    job_info = parse_json_from_llm_response(llm_response)
    if not job_info or not all(k in job_info for k in ["title", "company", "location", "search_keywords", "summary"]):
        print("❌ Failed to extract complete job information from LLM response.")
        return {}
    return job_info

# --- Profile Discovery & Enhancement ---

def fetch_linkedin_profile_data(url: str) -> Dict[str, Any]:
    """Fetches enriched LinkedIn data for a profile URL using RapidAPI."""
    cache_key = f"linkedin_{urllib.parse.quote(url)}"
    cached_data = read_from_cache(cache_key)
    if cached_data:
        print(f"⚡️ Using cached LinkedIn data for {url}")
        return cached_data

    print(f"⬇️ Fetching LinkedIn data for {url} from RapidAPI...")
    try:
        encoded_url = urllib.parse.quote(url)
        conn = http.client.HTTPSConnection(RAPIDAPI_HOST)
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": RAPIDAPI_HOST
        }
        endpoint = f"/get-linkedin-profile-by-salesnavurl?linkedin_url={encoded_url}&include_skills=true"
        conn.request("GET", endpoint, headers=headers)
        response = conn.getresponse()
        data = json.loads(response.read().decode())

        if response.status != 200:
            print(f"❌ RapidAPI error {response.status}: {data.get('message', 'Unknown error')}")
            return {"linkedin_url": url, "error": data.get('message', 'API error')}

        profile = data.get("data")
        if not isinstance(profile, dict):
            raise ValueError(f"Invalid RapidAPI response format: {data}")

        profile["linkedin_url"] = profile.get("linkedin_url", url) # Ensure URL is present
        profile["full_name"] = profile.get("full_name") or profile.get("name") # Harmonize name key
        write_to_cache(cache_key, profile)
        return profile
    except Exception as e:
        print(f"❌ RapidAPI fetch failed for {url}: {e}")
        return {"linkedin_url": url, "error": str(e)}

def enhance_profile_with_Google_Search(profile: Dict) -> Dict:
    """Enhances a candidate's profile with additional social media or personal website links found via Google Search."""
    if not profile.get("full_name"):
        return profile

    print(f"🔎 Enhancing profile for {profile.get('full_name', 'Unknown')} with Google Search...")
    query = f'{profile["full_name"]} github OR twitter OR personal website'
    params = {"q": query, "engine": "google", "num": 5, "api_key": SERPAPI_API_KEY}
    
    try:
        results = GoogleSearch(params).get_dict().get("organic_results", [])
        for res in results:
            link = res.get("link", "")
            if "github.com" in link and not profile.get("github_url"):
                profile["github_url"] = link
            elif "x.com" in link or "twitter.com" in link and not profile.get("twitter_url"):
                profile["twitter_url"] = link
            elif not profile.get("personal_website") and "linkedin" not in link:
                # Basic check to avoid re-adding LinkedIn profile as personal website
                profile["personal_website"] = link
    except Exception as e:
        print(f"❌ Google Search enhancement failed for {profile.get('full_name')}: {e}")
    return profile

def search_for_linkedin_profiles(query: str, max_results: int = 10) -> List[Dict]:
    """Performs a Google search for LinkedIn profiles and enriches them."""
    print(f"🔍 Searching Google for profiles: {query} (max {max_results} results)")
    params = {
        "q": query,
        "engine": "google",
        "num": max_results,
        "api_key": SERPAPI_API_KEY
    }
    
    profiles: List[Dict] = []
    seen_urls = set()

    try:
        results = GoogleSearch(params).get_dict()
        for res in results.get("organic_results", []):
            url = res.get("link", "")
            if "linkedin.com/in/" in url and url not in seen_urls:
                seen_urls.add(url)
                # Fetching and enhancing can be slow, but done sequentially here
                # Consider using ThreadPoolExecutor for concurrent profile fetching if needed
                profile = fetch_linkedin_profile_data(url)
                if not profile.get("error"):
                    enhanced_profile = enhance_profile_with_Google_Search(profile)
                    profiles.append(enhanced_profile)
                else:
                    print(f"Skipping profile due to error: {profile.get('error')} for {url}")
                time.sleep(1.2)  # Respect API rate limits and avoid being blocked
    except Exception as e:
        print(f"❌ SerpApi search failed: {e}")
    
    print(f"✅ Found and enriched {len(profiles)} unique profiles.")
    return profiles

# --- Scoring & Confidence ---

def calculate_confidence_score(candidate: Dict) -> float:
    """
    Calculates a confidence score based on the completeness of the candidate's data.
    Score ranges from 0.0 to 1.0.
    """
    score = 0.0
    # Base score for having a LinkedIn profile
    if candidate.get("linkedin_url") and not candidate.get("error"):
        score += 0.4 # Significant base score

    # Add points for specific key data points
    if candidate.get("full_name") and candidate.get("headline"): score += 0.1
    if candidate.get("experience"): score += 0.15 # Experience is crucial
    if candidate.get("skills"): score += 0.1 # Skills are also important
    if candidate.get("education"): score += 0.05 # Education
    if candidate.get("github_url"): score += 0.1
    if candidate.get("personal_website"): score += 0.05

    return round(min(1.0, score), 2) # Cap at 1.0 and round

def score_candidate(candidate: Dict, job: Dict) -> Dict:
    """Evaluates a candidate against job requirements using the LLM and assigns a fit score."""
    prompt = f"""
    You are an expert recruiter evaluating a candidate for a specific job role.
    Analyze the provided job description and candidate's profile to assess their fit.
    Return your analysis as a strict JSON object with the following keys and values:

    'education_tier': (string) Categorize candidate's education: 'elite' (e.g., Ivy League, top CS schools like MIT, Stanford, CMU, UIUC), 'strong' (reputable universities), 'standard' (others).
    'company_tier': (string) Categorize previous companies: 'top_tech' (FAANG, top startups), 'relevant_industry' (well-known companies in the specific industry), 'general' (others).
    'skills_overlap': (string) Assess skill match: 'perfect' (most key skills align), 'strong' (good overlap, some gaps), 'moderate' (some overlap, significant gaps), 'low' (minimal overlap).
    'location_match': (boolean) True if candidate's stated location or willingness to relocate matches the job location, False otherwise.
    'has_clear_progression': (boolean) True if candidate's career path shows clear progression and growth, False otherwise.
    'tenure_stability': (string) Assess job tenure: 'stable' (long tenure at most roles), 'moderate' (some shorter stints), 'varied' (frequent job changes).

    Job Description:\n{json.dumps(job, indent=2)}
    Candidate Profile:\n{json.dumps(candidate, indent=2)}
    """
    print(f"🤖 Scoring candidate: {candidate.get('full_name', 'Unknown')}")
    llm_response = get_llm_response(prompt)
    analysis = parse_json_from_llm_response(llm_response)

    # Default scores if analysis fails or is incomplete
    scores_breakdown = {
        "education": 6.0,
        "trajectory": 6.0,
        "company": 6.0,
        "skills": 6.0,
        "location": 6.0,
        "tenure": 6.0
    }

    # Assign scores based on LLM analysis
    if analysis:
        if analysis.get("education_tier") == "elite": scores_breakdown["education"] = 9.5
        elif analysis.get("education_tier") == "strong": scores_breakdown["education"] = 7.5

        if analysis.get("has_clear_progression"): scores_breakdown["trajectory"] = 9.0

        if analysis.get("company_tier") == "top_tech": scores_breakdown["company"] = 9.5
        elif analysis.get("company_tier") == "relevant_industry": scores_breakdown["company"] = 7.5

        if analysis.get("skills_overlap") == "perfect": scores_breakdown["skills"] = 9.5
        elif analysis.get("skills_overlap") == "strong": scores_breakdown["skills"] = 8.0
        elif analysis.get("skills_overlap") == "moderate": scores_breakdown["skills"] = 7.0

        if analysis.get("location_match"): scores_breakdown["location"] = 10.0

        if analysis.get("tenure_stability") == "stable": scores_breakdown["tenure"] = 9.0
        elif analysis.get("tenure_stability") == "moderate": scores_breakdown["tenure"] = 7.5

    # Calculate overall fit score with weights
    fit_score = round(
        scores_breakdown["education"] * 0.2 +
        scores_breakdown["trajectory"] * 0.2 +
        scores_breakdown["company"] * 0.15 +
        scores_breakdown["skills"] * 0.25 +
        scores_breakdown["location"] * 0.10 +
        scores_breakdown["tenure"] * 0.10,
        2
    )

    candidate["fit_score"] = fit_score
    candidate["score_breakdown"] = scores_breakdown
    candidate["confidence_score"] = calculate_confidence_score(candidate)
    return candidate

# --- Outreach ---

def generate_outreach_message(candidate: Dict, job: Dict) -> Dict[str, str]:
    """Generates a personalized LinkedIn outreach message and highlights for the candidate."""
    prompt = f"""
    You are a professional recruiter drafting an initial LinkedIn outreach message.
    Create a concise, engaging message (under 120 words) and list the key reasons why this candidate is a strong fit.
    Return your output as a strict JSON object with two keys:
    'message': (string) The LinkedIn outreach message.
    'match_highlights': (list of strings) Bullet points summarizing the main reasons for the candidate's fit.

    Role Description:\n{json.dumps(job, indent=2)}
    Candidate Profile:\n{json.dumps(candidate, indent=2)}
    """
    print(f"🤖 Generating outreach for {candidate.get('full_name', 'Unknown')}")
    llm_response = get_llm_response(prompt)
    outreach_data = parse_json_from_llm_response(llm_response)

    if not outreach_data or not all(k in outreach_data for k in ["message", "match_highlights"]):
        print("❌ Failed to generate complete outreach message from LLM response.")
        return {"message": "Could not generate a personalized message at this time.", "match_highlights": []}
    return outreach_data

# --- Main Pipeline ---

def process_single_job_description(job_description_text: str, num_candidates_to_find: int = 5) -> Dict[str, Any]:
    """
    Processes a single job description to find, score, and generate outreach for candidates.
    """
    job = analyze_job_description(job_description_text)
    if not job:
        print("🛑 Skipping job due to parsing failure.")
        return {"job_id": "N/A", "candidates_found": 0, "top_candidates": []}

    job_id = f"{job['company'].lower().replace(' ','-')}-{job['title'].lower().replace(' ','-')}"
    print(f"\n--- Processing Job: {job['title']} at {job['company']} ---")

    search_query = f'site:linkedin.com/in/ {" ".join(job["search_keywords"])} "{job["location"]}"'
    found_profiles = search_for_linkedin_profiles(search_query, max_results=num_candidates_to_find)

    if not found_profiles:
        print(f"⚠️ No suitable LinkedIn profiles found for {job['title']}.")
        return {"job_id": job_id, "candidates_found": 0, "top_candidates": []}

    print(f"📊 Scoring and generating outreach for {len(found_profiles)} candidates...")
    processed_candidates: List[Dict] = []
    for profile in found_profiles:
        if not profile.get("error"): # Only process valid profiles
            scored_candidate = score_candidate(profile, job)
            outreach_details = generate_outreach_message(scored_candidate, job)

            processed_candidates.append({
                "name": scored_candidate.get("full_name") or extract_name_from_linkedin_url(scored_candidate.get("linkedin_url")),
                "linkedin_url": scored_candidate.get("linkedin_url"),
                "fit_score": scored_candidate.get("fit_score"),
                "confidence_score": scored_candidate.get("confidence_score"),
                "score_breakdown": scored_candidate.get("score_breakdown"),
                **outreach_details # Merge outreach message and highlights
            })

    # Sort candidates by fit score in descending order
    top_candidates = sorted(processed_candidates, key=lambda x: x.get("fit_score", 0), reverse=True)

    print(f"--- Finished processing {job['title']} ---")
    return {
        "job_id": job_id,
        "candidates_found": len(top_candidates),
        "top_candidates": top_candidates
    }

def run_sourcing_agent_batch(job_descriptions: List[str], num_candidates_to_find: int = 5):
    """
    Runs the AI sourcing agent for a batch of job descriptions in parallel.
    """
    print(f"🚀 Starting batch processing for {len(job_descriptions)} jobs...")
    
    # Use ThreadPoolExecutor for concurrent job processing
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor: # Adjust max_workers as needed
        results = list(executor.map(
            lambda jd: process_single_job_description(jd, num_candidates_to_find),
            job_descriptions
        ))

    print("\n✅ Batch pipeline complete!")
    print(json.dumps(results, indent=2))
    return results

# --- Entry Point ---
if __name__ == "__main__":
    example_job_texts = [
        """
        Software Engineer, ML Research
        Windsurf • Full Time • Mountain View, CA • $140,000 – $300,000 + Equity

        Windsurf (formerly Codeium) is a Forbes AI 50 company building developer tools through AI.
        Roles include: training LLMs, designing experiments, converting research into product features.
        Requirements: 2+ years engineering, top CS school (MIT, Stanford, CMU, UIUC), proven ML experience, curiosity about codegen, on-site at Mountain View, CA.
        """,
        """
        Senior Product Manager, Growth
        Connectly • Full Time • San Francisco, CA • $160,000 - $220,000

        Connectly is a leader in conversational AI, helping businesses connect with customers.
        We are looking for a PM to drive user acquisition and retention.
        Requirements: 5+ years of PM experience, experience with A/B testing, data-driven, strong communication skills.
        """
    ]
    run_sourcing_agent_batch(example_job_texts, num_candidates_to_find=3)