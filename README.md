# ARK2.0

ARK (Automated Resource Knowledgebase) revolutionizes resource management via automation. Using advanced algorithms, it streamlines collection, organization, and access to resource data, facilitating efficient decision-making.

tl;dr. It'll be an open source interface for a local LLM app store utilizing long term memory for personalized requests. We are primarily focusing on targeting MIT students for the time being.

(TODO: rewrite this introduction.)

# Languages and dependencies

NOTE: This is not a complete list.

* Python
* LangChain (deprecated, we are trying to eliminate this for higher flexibility)
* Model Context Protocol
* HTTPX
* HuggingFace (what we currently use as our inference provider)
* Prettier
* Pydantic
* Radicale
* SQLite
* Svelte
* SvelteKit
* TypeScript
* @event-calendar/core

# File structure

(As of July 3, 2025; update as needed.)

This repo is currently a bit chaotic, but from a top-level point of view, here's each file or folder is for:

* `base_module/` for MCP- and database-related stuff
* `calendar_module/` (mostly empty)
* `config_module/` for YAML configuration files
* `depricated/` for old, unused code
* `frontend/` is the tool's web frontend, implemented in Svelte and TypeScript; read the [folder readme](frontend/README.md) for more information
* `model_module/` for core LLM-related logic
* `radicale-test/` is for using [Radicale](https://radicale.org/v3.html) to make calendar changes; read the [folder readme](radicale_test/README.md) for more information
* `schemas/` for JSON schemas when communicating between frontend and backend
* `.gitignore` (standard)
* `.nvimlog` (empty)
* `README.md` (this very file!)
* `requirements.txt` (Python dependencies; rather bloated for now, but it'll have to do)

# Instructions

## Start Inference Engine (HF-TGI)
* cmd: bash model_module/hftgi_2.sh
* Note: Qwen 2.5 is what is currently in use 

## Test model_module (Model Class ARKOAI)
* cmd: python ArkModelRefactored.py

# Contributors + contact

| Name                  | Role           | GitHub username | Affiliation |
| --------------------  | -------------- | --------------- | ----------- |
| Nathaniel Morgan      | Project leader | nmorgan         | MIT         |
| Joshua Guo            | Frontend       | duck_master     | MIT         |
| Ilya Gulko            | Backend        | gulkily         | MIT         |
| Yeabkal Abeje         | (departed)     | Yebe-Abe        | MIT         |
| Jack Luo              | Backend        |                 |             |
| Bryce Roberts         | Backend        |                 |             | 

