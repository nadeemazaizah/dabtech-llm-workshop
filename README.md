## DabTech LLM Workshop

This repository contains the code and materials used in the Dabburiya Tech session **“Beyond Chatbots – Building the Next Generation of AI Agents.”**

The workshop is split into 5 parts, each demonstrating a different LLM-based application pattern.

- **Part 1 – Simple LLM Service**: A basic LLM app that uses a **fixed context** loaded from a file to answer generic questions about the Dabburiya community.
- **Part 2 – RAG with Profiles**: A **Retrieval-Augmented Generation (RAG)** example that uses **Dabburiya Tech members’ LinkedIn-like profiles** as the knowledge base.
- **Part 3 – GraphRAG with LightRAG**: Uses **LightRAG / GraphRAG** over the same members’ profiles to build a knowledge graph and answer more complex questions about members and their relationships.
- **Part 4 – Text-to-SQL on Members Stats**: Demonstrates how to turn natural-language questions into **SQL queries** over a members statistics dataset.
- **Part 5 – Agents with Tools (The Blue Alliance)**: Shows how to build an **agent** that answers questions about FRC robotics teams and games using tools / APIs, specifically **The Blue Alliance API** to fetch and summarize information about FRC teams.

The goal is to show a practical path from a simple context-based LLM service to more advanced RAG, GraphRAG, Text-to-SQL, and tool-using agents.

---

## 1. Prerequisites

Before you start:

- **Python**: 3.10+ installed on your machine.
- **Git**: To clone this repository.
- **GitHub Models access**: You need a **GitHub Models API token** (a fine-grained personal access token with **Models** permission).

> If you don’t have access yet, follow the **Getting GitHub Models Access** section below.

---

## 1.1 Getting GitHub Models Access

To call GitHub-hosted models from this workshop, you need a fine-grained personal access token with the **Models** permission.

Steps:

1. Sign in to your GitHub account.
2. Go to **Settings → Developer settings → Personal access tokens**.
3. Open the **Fine-grained tokens** tab.
4. Click **Generate new token**.
5. Give your token a descriptive name (for example: `llm-models`).
6. Leave **Repository access** set to **Public repositories**.
7. Click **+ Add permissions**.
8. In the **Select account permissions** list, choose **Models** and grant the required access.
9. Generate the token and **copy** it somewhere safe.

You will use this token as your LLM API key in `.env`:

```bash
OPENAI_API_KEY=your_copied_github_models_token_here
```

> Note: GitHub Models provides free access to popular LLMs, including **GPT-4o-mini**.

---

## 2. Clone the Repository

Open a terminal (PowerShell on Windows) and run:

```pwsh
git clone https://github.com/nadeemazaizah/dabtech-llm-workshop.git
cd dabtech-llm-workshop
```

---

## 3. Environment Variables & API Keys

This project uses environment variables to configure:

- **GitHub Models** (LLM API key and base URL)
- **The Blue Alliance API** (for Part 5 – Agents)

### 3.1. Create your `.env` file

In the project root, there should be a file named `.env.example`.

1. Copy it to `.env`:

```pwsh
Copy-Item .env.example .env
```

2. Open `.env` in your editor and set the following values:

```bash
OPENAI_API_KEY=your_github_models_token_here
OPENAI_API_BASE=https://models.inference.ai.azure.com

# For Part 5 – Agents / The Blue Alliance
TBA_KEY=your_the_blue_alliance_api_key_here
```

> **Note:** Make sure `.env` is **not committed** to Git (it should already be gitignored) because it contains secrets.

### 3.2. GitHub Models API key & base URL

You already created your GitHub Models token in **1.1 Getting GitHub Models Access** and stored it as `OPENAI_API_KEY` in `.env`.

For the base URL, use:

```bash
OPENAI_API_BASE=https://models.inference.ai.azure.com
```

Check GitHub Models documentation for any updates or region-specific endpoints.

### 3.3. Getting The Blue Alliance API key (for Part 5)

1. Go to **The Blue Alliance** website and create an account.
2. Navigate to the **API** section in your account.
3. Generate an API key.
4. Put that value in `TBA_KEY` inside `.env`.

---

## 4. Create and Activate a Virtual Environment

From the project root (`dabtech-llm-workshop`), run:

```pwsh
python -m venv venv

venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

If `Activate.ps1` is blocked by your PowerShell execution policy, you may need to run PowerShell as Administrator and temporarily relax the policy, for example:

```pwsh
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Then run `venv\Scripts\Activate.ps1` again.

---

## 5. Install Dependencies

Install all required Python packages using `requirements.txt`:

```pwsh
pip install -r requirements.txt
```

---

## 6. Project Structure

Simplified view of the relevant folders:

- `data/`
	- Community info, members stats (for Text-to-SQL), LanceDB vectors (for RAG), LightRAG storage (for GraphRAG), and profiles used in the different parts.
- `notebooks/`
	- Jupyter notebooks with demos (e.g. `Demo.ipynb`, `test_notebook.ipynb`).
- `src/`
	- `Part1_Simple_LLM/`
	- `Part2_RAG/`
	- `Part3_GraphRAG/`
	- `Part4_Text2SQL/`
	- `Part5_Agent/`
	- `Chainlit_App/` (if you use Chainlit for UI demos)

Each `PartX_*` folder contains the code for that specific part covered in the session.

---

## 7. Running Each Part

Below is a high-level guide for running each part.

### 7.1 Part 1 – Simple LLM Service

**Goal:** Simple LLM service with a **fixed context** loaded from a file that answers generic questions about the Dabburiya community.

Typical steps:

1. Ensure `.env` has your GitHub Models credentials.
2. Navigate to the Part 1 folder:

```pwsh
cd src\Part1_Simple_LLM
```

3. Run the simple LLM chat script:

```pwsh
python simple_llm_chat.py
```

This script loads a text file from `data/` containing generic community info and uses it to answer questions.

### 7.2 Part 2 – RAG with Members Profiles

**Goal:** Demonstrate **Retrieval-Augmented Generation (RAG)** using Dabburiya Tech members’ profiles (from LinkedIn-like text files in `data/profiles_examples` and `data/profiles_full`).

Steps:

1. Ensure `.env` has GitHub Models credentials.
2. Navigate to Part 2:

```pwsh
cd src\Part2_RAG
```

3. First, build the RAG index:

```pwsh
python rag_indexing.py
```

4. Then run the RAG chat script:

```pwsh
python rag_chat.py
```

This part will:

- Index profiles from the `data/profiles_*` directories.
- Use a vector database / embeddings to retrieve relevant profiles.
- Feed the retrieved context plus your question into the LLM.

Try questions like:

- "Who in Daburiya Tech community has experience with machine learning?"

### 7.3 Part 3 – GraphRAG with LightRAG

**Goal:** Use **GraphRAG** via **LightRAG** to build a knowledge graph over the same members profiles and answer complex questions.

Steps:

1. Ensure `.env` has GitHub Models credentials.
2. Navigate to Part 3:

```pwsh
cd src\Part3_GraphRAG
```

3. First, build the GraphRAG index:

```pwsh
python graphrag_indexing.py
```

4. Then run the GraphRAG chat script:

```pwsh
python graphrag_chat.py
```

This part will:

- Use LightRAG with the profiles data.
- Build / load a graph (see `data/lightrag_storage` and `data/lancedb`).
- Answer questions about members and relationships (e.g., collaborations, skills graphs, etc.).

Try questions like:

- "Who in Daburiya Tech community has experience with machine learning?"

### 7.4 Part 4 – Text-to-SQL on Members Stats

**Goal:** Show how to convert natural-language questions into **SQL queries** over a members statistics dataset.

The dataset is in `data/members_stats.csv`.

Steps:

1. Ensure `.env` has GitHub Models credentials.
2. Navigate to Part 4:

```pwsh
cd src\Part4_Text2SQL
```

3. Run the Text-to-SQL chat script:

```pwsh
python text_to_sql_chat.py
```

Typical flow:

- User asks a question (e.g., "How many members work in data?" or "Average years of experience per domain?").
- The LLM generates a SQL query.
- The app executes the SQL against a database or in-memory table created from `members_stats.csv`.
- The result is returned to the user.

### 7.5 Part 5 – Agent with Tools (The Blue Alliance)

**Goal:** Demonstrate an **agent** that uses tools / APIs. In this part, the agent calls **The Blue Alliance API** to answer questions about FRC teams.

Steps:

1. Ensure `.env` contains:

	 - `GITHUB_MODELS_API_KEY`
	 - `GITHUB_MODELS_BASE_URL`
	 - `BLUE_ALLIANCE_API_KEY`

2. Navigate to Part 5:

```pwsh
cd src\Part5_Agent
```

3. Run the FRC agent script:

```pwsh
python frc_agent.py
```

The agent:

- Interprets your question.
- Decides when to call The Blue Alliance API tools.
- Calls the tools (using your `BLUE_ALLIANCE_API_KEY`).
- Combines tool outputs with LLM reasoning to generate a final answer.

Try questions like:

- "Tell me about FRC team 254."
- "Which teams performed best in the last season?"

---

## 8. Using the Chainlit App (Optional)

If this repo includes a `Chainlit_App` folder, you can run an interactive UI for some parts using **Chainlit**.

Typical steps:

```pwsh
cd src\Chainlit_App
chainlit run app.py
```

Check `src/Chainlit_App/chainlit.md` for exact commands and configuration.

---