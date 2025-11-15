from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# MASTER ENGINE FOR SUGGEST MODE
MASTER_ENGINE = """
You are the Prompt Recommendation & Generation Engine.

Your job:
1. Analyze the user's idea.
2. Determine the best prompt type(s) from this list:
- System
- Instruction
- Zero-Shot
- Few-Shot
- Context / RAG
- Chain-of-Thought
- Self-Consistency
- Self-Critique / Reflection
- ReAct
- Tool-Use
- Planning
- Verification
- Template-Based
- Constraint-Based
- Agentic
3. Explain why (1 sentence).
4. Generate the final optimized prompt.

FORMAT YOUR OUTPUT EXACTLY LIKE THIS:

Recommended Prompt Type(s):
<type>

Short Explanation:
<sentence>

Final Generated Prompt:
<prompt>
"""

# GENERATOR ENGINE
PROMPT_GENERATOR = """
You are the Prompt Builder AI.

Generate a perfect {ptype} prompt.

User Goal:
{goal}

Examples:
{examples}

Context:
{context}

Constraints:
{constraints}

Tools:
{tools}

OUTPUT:
Only the final prompt.
No commentary.
"""


# ---- OPENAI WRAPPER ----
def chat(message):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content


# ---- ROUTES ----
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.json
    idea = data.get("idea", "")

    final = chat(MASTER_ENGINE + "\n\nUser Idea:\n" + idea)
    return jsonify({"result": final})


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json

    ptype = data.get("ptype", "")
    goal = data.get("goal", "")
    examples = data.get("examples", "")
    context = data.get("context", "")
    constraints = data.get("constraints", "")
    tools = data.get("tools", "")

    prompt = PROMPT_GENERATOR.format(
        ptype=ptype,
        goal=goal,
        examples=examples,
        context=context,
        constraints=constraints,
        tools=tools
    )

    final = chat(prompt)
    return jsonify({"result": final})


# ---- RUN ----
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
