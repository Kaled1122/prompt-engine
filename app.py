from flask import Flask, request, jsonify, render_template
import openai
import os

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")


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

Your output MUST follow this exact format:

Recommended Prompt Type(s):
<type>

Short Explanation:
<sentence>

Final Generated Prompt:
<prompt>
"""


PROMPT_GENERATOR = """
You are the Prompt Builder AI.
Generate a perfect {ptype} prompt based on the user's goal.

User Goal: {goal}
Examples: {examples}
Context: {context}
Constraints: {constraints}
Tools: {tools}

Output ONLY the final prompt. No explanation.
"""


def chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
