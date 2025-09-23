# ARKOS

ARK (Automated Resource Knowledgebase) revolutionizes resource management via automation. Using advanced algorithms, it streamlines collection, organization, and access to resource data, facilitating efficient decision-making.

tl;dr. It'll be an open source interface for a local LLM agent building utilizing long term memory for personalized requests. 

## Languages and dependencies

The entire codebase is in Python, except for a few shell scripts. We use the following four dependencies:

* `openai` (needed to standardize inference engine communication; we do not use the OpenAI API though!)
* `pyyaml`
* `pydantic` for defining schemas
* `requests`
* `mcp-use`

## File structure

(As of September 11, 2025.)

This repo is rather chaotic, but from a top-level point of view, here's each file or folder is for:


* `base_module/` for main interface
* `config_module/` for YAML configuration files
* `model_module/` for core LLM-inference logic
* `agent_module/` for agentic structure 
* `state_module/` for defining agent state graphs 
* `tool_module/` for MCP compatibility 
* `memory_module/` for long term memory and context management
* `schemas/` for JSON schemas when communicating between frontend and backend
* `state_module/`
* `tool_module/`
* `.gitignore`
* `README.md` (this very file!)
* `requirements.txt` (Python dependencies)

## Instructions

### Start Inference Engine (SGLANG)

* Run latest SGLANG image
* cmd: bash model_module/run.sh
* Note: Qwen 2.5 is what is currently in use

### Test base_module

* cmd: python main_interface.py

## Contributors + contact

| Name                  | Role           | GitHub username | Affiliation   |
| --------------------  | -------------- | --------------- | --------------|
| Nathaniel Morgan      | Project leader | nmorgan         | MIT           |
| Joshua Guo            | Frontend       | duck_master     | MIT           |
| Ilya Gulko            | Backend        | gulkily         | MIT           |
| Yeabkal Abeje         | (departed)     | Yebe-Abe        | MIT           |
| Jack Luo              | Backend        | thejackluo      | Georgia Tech  |
| Bryce Roberts         | Backend        | BryceRoberts13  | MIT           |
