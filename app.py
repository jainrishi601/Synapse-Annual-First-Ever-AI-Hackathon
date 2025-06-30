import gradio as gr
from main2 import run_sourcing_agent_batch  # Replace with the actual file name (e.g., import main if saved as main.py)

def process_jobs_interface(job_descriptions_text, num_candidates):
    job_descriptions = [desc.strip() for desc in job_descriptions_text.strip().split("\n\n") if desc.strip()]
    results = run_sourcing_agent_batch(job_descriptions, num_candidates_to_find=num_candidates)

    output_str = ""
    for result in results:
        output_str += f"\n📌 **Job ID**: `{result['job_id']}`\n"
        output_str += f"👥 Candidates Found: {result['candidates_found']}\n"
        for idx, cand in enumerate(result["top_candidates"], 1):
            output_str += f"\n--- Candidate #{idx} ---\n"
            output_str += f"🔹 Name: {cand['name']}\n"
            output_str += f"🔗 LinkedIn: {cand['linkedin_url']}\n"
            output_str += f"📊 Fit Score: {cand['fit_score']} | 🤝 Confidence: {cand['confidence_score']}\n"
            output_str += f"🧠 Message:\n{cand['message']}\n"
            output_str += "✅ Match Highlights:\n"
            for h in cand["match_highlights"]:
                output_str += f"- {h}\n"
            output_str += "\n"
    return output_str.strip()

with gr.Blocks(title="AI Job Candidate Sourcing Agent") as demo:
    gr.Markdown("# 🤖 AI Candidate Sourcing Agent")
    gr.Markdown("Paste one or more job descriptions (separated by two newlines), and get scored and ranked LinkedIn candidates.")

    with gr.Row():
        job_input = gr.Textbox(lines=20, label="Job Descriptions", placeholder="Paste multiple job descriptions here, separated by two newlines (\\n\\n)...")
        candidate_slider = gr.Slider(1, 10, step=1, value=3, label="Number of Candidates per Job")

    run_button = gr.Button("🔍 Run Sourcing Pipeline")
    output_box = gr.Textbox(lines=30, label="Results", interactive=False)

    run_button.click(fn=process_jobs_interface, inputs=[job_input, candidate_slider], outputs=output_box)

if __name__ == "__main__":
    demo.launch()
