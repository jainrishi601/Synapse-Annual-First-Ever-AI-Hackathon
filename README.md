# AI-Powered Candidate Sourcing Agent 🤖

An intelligent candidate sourcing system that uses AI to find, evaluate, and generate personalized outreach for potential candidates based on job descriptions.

## 🎯 Overview

This project automates the candidate sourcing process by combining multiple AI services to:
- **Analyze job descriptions** using LLM to extract key requirements
- **Search for candidates** using Google Search API to find LinkedIn profiles
- **Enrich candidate data** using RapidAPI to get detailed profile information
- **Score candidates** using AI to evaluate fit based on multiple criteria
- **Generate personalized outreach** messages tailored to each candidate

## 🧠 AI Approach & Methodology

### Multi-Stage AI Pipeline

1. **Job Analysis Stage**
   - Uses DeepSeek-V3-0324 LLM to parse job descriptions
   - Extracts structured data: title, company, location, keywords
   - Generates optimized search queries for candidate discovery

2. **Candidate Discovery Stage**
   - Leverages Google Search API to find LinkedIn profiles
   - Uses intelligent search patterns: `site:linkedin.com/in/ + keywords + location`
   - Implements deduplication to avoid processing same profiles

3. **Profile Enhancement Stage**
   - Fetches detailed profile data via RapidAPI
   - Discovers additional social media links (GitHub, Twitter, personal websites)
   - Caches responses to minimize API costs and improve performance

4. **AI Scoring Stage**
   - Multi-dimensional evaluation using LLM analysis:
     - **Education Tier**: Elite institutions vs. standard universities
     - **Company Tier**: FAANG/top tech vs. relevant industry vs. general
     - **Skills Overlap**: Perfect, strong, moderate, or low match assessment
     - **Location Match**: Geographic compatibility evaluation
     - **Career Progression**: Growth trajectory analysis
     - **Tenure Stability**: Job stability and commitment assessment
   - Weighted scoring system with configurable weights
   - Confidence scoring based on data completeness

5. **Outreach Generation Stage**
   - Creates personalized LinkedIn messages using LLM
   - Highlights specific match points for each candidate
   - Maintains professional tone and engagement optimization

## 🚀 Key Features

### 🔍 Intelligent Job Analysis
- Extracts job title, company, location, and search keywords
- Generates optimized search queries for candidate discovery
- Provides concise role summaries

### 👥 Advanced Candidate Discovery
- Searches LinkedIn profiles using Google Search API
- Enriches profiles with additional data from RapidAPI
- Discovers GitHub, Twitter, and personal website links
- Implements intelligent caching to avoid duplicate API calls

### 📊 AI-Powered Candidate Scoring
- **Education Tier**: Elite (Ivy League, top CS schools) vs. standard
- **Company Tier**: Top tech (FAANG) vs. relevant industry vs. general
- **Skills Overlap**: Perfect, strong, moderate, or low match
- **Location Match**: Geographic compatibility assessment
- **Career Progression**: Clear growth trajectory evaluation
- **Tenure Stability**: Job stability and commitment analysis

### 💬 Personalized Outreach Generation
- Creates tailored LinkedIn outreach messages
- Highlights key match points for each candidate
- Maintains professional tone and engagement

### ⚡ Performance Optimizations
- Intelligent caching system (24-hour expiration)
- Concurrent processing for batch operations
- Rate limiting to respect API constraints
- Error handling and graceful degradation

## 🛠️ Technology Stack

- **AI/LLM**: Hugging Face Inference API (DeepSeek-V3-0324)
- **Search**: SerpAPI (Google Search)
- **Profile Data**: RapidAPI (LinkedIn Profile Data)
- **Language**: Python 3.8+
- **Key Libraries**: 
  - `google-generativeai` for AI interactions
  - `serpapi` for web search
  - `huggingface_hub` for LLM inference
  - `python-dotenv` for environment management

## 📋 Prerequisites

Before running this project, you'll need:

1. **API Keys** (see setup section below):
   - **Hugging Face API Token**: For LLM inference (DeepSeek-V3-0324 model)
   - **SerpAPI Key**: For Google Search to find LinkedIn profiles
   - **RapidAPI Key**: For LinkedIn profile data enrichment

2. **Python Environment**:
   - Python 3.8 or higher
   - pip package manager

## 🔑 API Setup Guide

### 1. Hugging Face API Token
1. Go to [Hugging Face](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token to your `.env` file

### 2. SerpAPI Key
1. Sign up at [SerpAPI](https://serpapi.com/)
2. Get your API key from the dashboard
3. Add to your `.env` file

### 3. RapidAPI Key
1. Sign up at [RapidAPI](https://rapidapi.com/)
2. Subscribe to "Fresh LinkedIn Profile Data" API
3. Get your API key from the dashboard
4. Add to your `.env` file

## 📁 Project Structure

```
synapseint/
├── hf_dir/
│   └── Candidate_searcher/
│       ├── main2.py              # Main application file
│       ├── app.py                # Alternative app interface
│       ├── requirements.txt      # Python dependencies
│       ├── README.md            # Project documentation
│       └── cache/               # Cached API responses
├── candidates_job_*.csv         # Generated candidate lists
├── sourcing_job_*.json          # Job processing results
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd synapseint
```

### 2. Set Up Environment Variables
Create a `.env` file in the project root with your API keys:

```env
HF_TOKEN=your_huggingface_token_here
SERPAPI_API_KEY=your_serpapi_key_here
RAPIDAPI_KEY=your_rapidapi_key_here
```

### 3. Install Dependencies
```bash
cd hf_dir/Candidate_searcher
pip install -r requirements.txt
```

### 4. Run the Sourcing Agent
```bash
python main2.py
```

The script will run with example job descriptions and demonstrate the full pipeline.

## 📖 Usage Examples

### Single Job Processing
```python
from main2 import process_single_job_description

job_description = """
Senior Machine Learning Engineer
TechCorp • Full Time • San Francisco, CA • $150,000 - $200,000

We're looking for an experienced ML engineer to join our AI team.
Requirements: 3+ years ML experience, Python, PyTorch, AWS, PhD preferred.
"""

result = process_single_job_description(job_description, num_candidates_to_find=5)
print(f"Found {result['candidates_found']} candidates")
```

### Batch Processing
```python
from main2 import run_sourcing_agent_batch

job_descriptions = [
    "Job description 1...",
    "Job description 2...",
    "Job description 3..."
]

results = run_sourcing_agent_batch(job_descriptions, num_candidates_to_find=3)
```

## 📊 Output Format

The system returns structured data for each candidate:

```json
{
  "job_id": "company-job-title",
  "candidates_found": 5,
  "top_candidates": [
    {
      "name": "John Doe",
      "linkedin_url": "https://linkedin.com/in/johndoe",
      "fit_score": 8.5,
      "confidence_score": 0.85,
      "score_breakdown": {
        "education": 9.5,
        "trajectory": 8.0,
        "company": 9.5,
        "skills": 8.5,
        "location": 10.0,
        "tenure": 7.5
      },
      "message": "Personalized outreach message...",
      "match_highlights": [
        "Strong ML background with 5+ years experience",
        "Graduated from Stanford with CS degree",
        "Previous experience at Google and Meta"
      ]
    }
  ]
}
```

## 🔧 Configuration

### Cache Settings
- **Cache Directory**: `cache/` (auto-created)
- **Expiration**: 24 hours
- **Format**: JSON with timestamps

### Rate Limiting
- **Google Search**: 1.2 seconds between requests
- **RapidAPI**: Respects API limits
- **LLM Calls**: No built-in rate limiting (depends on provider)

### Scoring Weights
- **Education**: 20%
- **Career Trajectory**: 20%
- **Company Experience**: 15%
- **Skills Match**: 25%
- **Location**: 10%
- **Tenure Stability**: 10%

## 🎯 Use Cases

### For Recruiters
- **High-Volume Hiring**: Process multiple job descriptions simultaneously
- **Passive Candidate Sourcing**: Find candidates not actively job searching
- **Quality Assessment**: AI-powered candidate evaluation
- **Outreach Automation**: Generate personalized messages at scale

### For HR Teams
- **Talent Pipeline Building**: Continuously source candidates for future roles
- **Market Research**: Understand candidate landscape for specific roles
- **Competitive Analysis**: Identify where top talent works

### For Startups
- **Cost-Effective Sourcing**: Reduce expensive recruiter time
- **Scalable Growth**: Handle hiring needs as company scales
- **Data-Driven Decisions**: Make informed hiring choices

## 🔒 Privacy & Ethics

- **Data Usage**: Only processes publicly available LinkedIn profiles
- **API Compliance**: Respects all API rate limits and terms of service
- **Caching**: Stores data locally with expiration for efficiency
- **Transparency**: Clear logging of all operations

## 🚨 Limitations

- **API Dependencies**: Requires active API keys and internet connection
- **Data Quality**: Depends on accuracy of LinkedIn profile data
- **Rate Limits**: Processing speed limited by API constraints
- **Profile Availability**: Only finds publicly accessible LinkedIn profiles
- **Cost**: API calls incur charges (SerpAPI, RapidAPI, Hugging Face)
- **Accuracy**: AI scoring is probabilistic and should be validated by humans

## 🔧 Troubleshooting

### Common Issues

**"LLM service not available"**
- Check your Hugging Face API token is valid
- Ensure you have sufficient credits/quota
- Verify internet connection

**"No LinkedIn profiles found"**
- Check SerpAPI key and quota
- Try different search keywords
- Verify job location format

**"RapidAPI fetch failed"**
- Verify RapidAPI subscription is active
- Check API key permissions
- Ensure LinkedIn URLs are valid

**Cache Issues**
- Delete `cache/` directory to clear all cached data
- Check file permissions in cache directory

### Performance Tips

- **Batch Processing**: Use `run_sourcing_agent_batch()` for multiple jobs
- **Caching**: System automatically caches responses for 24 hours
- **Rate Limiting**: Built-in delays prevent API throttling
- **Concurrent Processing**: Uses ThreadPoolExecutor for parallel execution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for providing the LLM inference infrastructure
- **SerpAPI** for Google Search capabilities
- **RapidAPI** for LinkedIn profile data access
- **DeepSeek AI** for the powerful language model

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the code comments
- Review the example usage in `main2.py`

---

**Happy Sourcing! 🎉**
