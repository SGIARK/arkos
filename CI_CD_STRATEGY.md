# ARKOS CI/CD Strategy & Implementation Guide

**Project:** ARKOS (Automated Resource Knowledgebase Operating System)
**Context:** Open-source MIT student project, Python-based local LLM agent framework
**Current State:** No existing CI/CD infrastructure
**Prepared by:** DevOps Team
**Date:** November 2025

---

## Executive Summary

This document provides comprehensive answers to CI/CD, testing, collaboration, and developer experience questions for the ARKOS project. Recommendations are based on thorough analysis of the current codebase and tailored for an open-source, student-led project with limited resources.

**Key Findings:**
- No CI/CD pipeline currently exists
- Minimal test coverage (2 test files only)
- No containerization of the application
- Missing dependency management
- Strong foundation for implementing CI/CD due to modular architecture

---

## 1. CI/CD Pipeline Strategy

### 1.1 Target Deployment Frequency

**Recommendation: Weekly releases with daily development builds**

**Rationale:**
- **Current Project Stage:** Early development (v0.x), features are still being defined
- **Team Composition:** MIT students with academic schedules - weekly releases align with sprint cycles
- **Infrastructure Complexity:** Requires GPU-backed SGLANG deployment - slower deployment cadence reduces operational burden
- **User Base:** Early adopters and contributors who value stability over rapid updates

**Deployment Cadence:**
- **Development Branch (`main`):** Automated deployment to staging on every merge (continuous deployment to staging)
- **Staging Environment:** Daily automated builds for testing
- **Production Releases:** Weekly tagged releases (e.g., `v0.1.0`, `v0.1.1`) deployed on Fridays
- **Hotfixes:** On-demand for critical bugs (bypass weekly schedule)

**Implementation:**
```yaml
# .github/workflows/deploy-staging.yml
on:
  push:
    branches: [main]

# .github/workflows/deploy-production.yml
on:
  release:
    types: [published]
```

---

### 1.2 Deployment Approval Process

**Recommendation: Semi-automatic with manual approval gates**

**Strategy:**

| Environment | Trigger | Approval Required | Approvers |
|------------|---------|-------------------|-----------|
| **Development** | Every PR merge to `main` | ❌ No (automatic) | N/A |
| **Staging** | Every commit to `main` | ❌ No (automatic) | N/A |
| **Production** | Release tag created | ✅ Yes (manual gate) | 2 maintainers |

**Approval Workflow:**
```yaml
# .github/workflows/deploy-production.yml
jobs:
  deploy-production:
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://arkos.mit.edu  # If applicable
    steps:
      - name: Require manual approval
        uses: trstringer/manual-approval@v1
        with:
          secret: ${{ github.TOKEN }}
          approvers: maintainer-1,maintainer-2
          minimum-approvals: 2
```

**Rationale:**
- **Automatic staging:** Fast feedback loop for developers, enables continuous testing
- **Manual production gate:** Prevents accidental deployments, allows final review of release notes
- **Dual approval requirement:** Reduces risk of single-person errors in student team
- **No manual dev approval:** Maintains developer velocity for experimental features

---

### 1.3 Environment Architecture

**Recommendation: 3 environments (Dev, Staging, Production)**

**Environment Structure:**

```
┌─────────────────────────────────────────────────────────────┐
│  DEVELOPMENT ENVIRONMENT                                     │
│  - Branch: feature/* branches                                │
│  - Deployment: Local developer machines only                 │
│  - SGLANG: Docker container per developer                    │
│  - Config: config_module/dev.yaml                           │
│  - Data: Ephemeral (reset frequently)                        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGING ENVIRONMENT                                         │
│  - Branch: main                                              │
│  - Deployment: Automated on merge to main                    │
│  - SGLANG: Shared GPU instance (e.g., MIT server)           │
│  - Config: config_module/staging.yaml                       │
│  - Data: Persistent, seeded with test data                   │
│  - URL: staging.arkos.internal (if applicable)               │
│  - Purpose: Integration testing, demo environment            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PRODUCTION ENVIRONMENT                                      │
│  - Branch: Release tags (v0.1.0, v0.2.0, etc.)              │
│  - Deployment: Manual approval required                      │
│  - SGLANG: Dedicated GPU instance with redundancy           │
│  - Config: config_module/production.yaml                    │
│  - Data: Persistent, backed up daily                         │
│  - URL: arkos.mit.edu or GitHub releases (if self-hosted)   │
│  - Purpose: Stable releases for end users                    │
└─────────────────────────────────────────────────────────────┘
```

**Why NOT QA environment:**
- **Resource constraints:** GPU instances are expensive, three environments are sufficient
- **Staging serves QA purpose:** MIT students can perform acceptance testing in staging
- **Project maturity:** Pre-1.0 software doesn't yet need separate QA environment
- **Can add later:** Easy to insert QA environment when project scales

**Environment Configuration:**

Create environment-specific YAML files in `config_module/`:

```yaml
# config_module/staging.yaml
environment: staging
model:
  url: "http://staging-sglang.mit.edu:30000/v1"
  model_name: "Qwen/Qwen2.5-7B-Instruct"
memory:
  storage_path: "/data/staging/memory.csv"
  backup_enabled: true
logging:
  level: DEBUG
  output: "/var/log/arkos/staging.log"
telemetry:
  enabled: true
  endpoint: "http://metrics.mit.edu/arkos"

# config_module/production.yaml
environment: production
model:
  url: "http://prod-sglang.mit.edu:30000/v1"
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  timeout: 60
  retry_attempts: 3
memory:
  storage_path: "/data/production/memory.csv"
  backup_enabled: true
  backup_frequency: "hourly"
logging:
  level: INFO
  output: "/var/log/arkos/production.log"
  rotation: "100MB"
telemetry:
  enabled: true
  endpoint: "http://metrics.mit.edu/arkos"
  sample_rate: 1.0
```

---

### 1.4 Production Deployment Authorization

**Recommendation: Tiered access control with explicit ownership**

**Access Levels:**

| Role | Can Trigger Production Deploy | Can Approve Production Deploy | Can Rollback |
|------|------------------------------|-------------------------------|--------------|
| **Core Maintainers** (2-3 people) | ✅ Yes (create release tag) | ✅ Yes (approve workflow) | ✅ Yes |
| **Contributors** (MIT students) | ❌ No | ❌ No | ❌ No |
| **DevOps Team** (you) | ✅ Yes | ✅ Yes | ✅ Yes |
| **External Contributors** | ❌ No | ❌ No | ❌ No |

**Implementation:**

1. **GitHub Repository Settings:**
   - Enable branch protection on `main`
   - Require pull request reviews (2 approvals)
   - Require status checks to pass
   - Restrict who can push to `main` (maintainers only)
   - Enable tag protection for `v*` tags

2. **GitHub Environments:**
```yaml
# Repository Settings → Environments → production
Environment name: production
Protection rules:
  ✅ Required reviewers: maintainer-1, maintainer-2, devops-lead
  ✅ Wait timer: 5 minutes (cooling-off period)
  ✅ Deployment branches: Only release tags (v*)
Environment secrets:
  - PRODUCTION_SSH_KEY
  - SGLANG_API_KEY
  - MONITORING_TOKEN
```

3. **Release Tag Workflow:**
```bash
# Only maintainers can perform this
git tag -a v0.1.0 -m "Release v0.1.0: Initial public release"
git push origin v0.1.0

# This triggers .github/workflows/deploy-production.yml
# which requires manual approval before deploying
```

**Rationale:**
- **Prevents accidental deployments:** Students learning the system can't accidentally push to production
- **Maintains velocity:** Contributors can still merge to staging automatically
- **Clear accountability:** Release tags are signed and attributed to specific maintainers
- **Emergency access:** DevOps team can override in critical situations

---

### 1.5 Automated Rollback Capabilities

**Recommendation: Yes, implement automated rollback with manual trigger**

**Rollback Strategy:**

**1. Automated Health Checks (Post-Deployment):**
```yaml
# .github/workflows/deploy-production.yml
jobs:
  deploy:
    # ... deployment steps ...

  health-check:
    needs: deploy
    runs-on: ubuntu-latest
    steps:
      - name: Smoke Test - Agent Initialization
        run: |
          curl -f http://production.arkos.mit.edu/health || exit 1

      - name: Smoke Test - SGLANG Connectivity
        run: |
          curl -f http://prod-sglang.mit.edu:30000/v1/models || exit 1

      - name: Smoke Test - Memory Operations
        run: |
          python scripts/test_memory_write.py --env production || exit 1

      - name: Auto-rollback on failure
        if: failure()
        run: |
          echo "Health checks failed, initiating rollback"
          ./scripts/rollback.sh ${{ github.event.before }}
```

**2. Manual Rollback Workflow:**
```yaml
# .github/workflows/rollback-production.yml
name: Rollback Production
on:
  workflow_dispatch:
    inputs:
      target_version:
        description: 'Version to rollback to (e.g., v0.1.0)'
        required: true
        type: string

jobs:
  rollback:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Validate target version exists
        run: |
          git fetch --tags
          git tag | grep -q "^${{ inputs.target_version }}$" || exit 1

      - name: Checkout target version
        uses: actions/checkout@v4
        with:
          ref: ${{ inputs.target_version }}

      - name: Deploy previous version
        run: ./scripts/deploy.sh ${{ inputs.target_version }}

      - name: Verify rollback
        run: ./scripts/health-check.sh

      - name: Notify team
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚨 Production rolled back to ${{ inputs.target_version }}"
            }
```

**3. Rollback Script (`scripts/rollback.sh`):**
```bash
#!/bin/bash
set -e

TARGET_VERSION=$1
BACKUP_DIR="/data/production/backups"

echo "Rolling back to version: $TARGET_VERSION"

# Stop current application
docker-compose -f production/docker-compose.yml down

# Restore previous configuration
cp "$BACKUP_DIR/$TARGET_VERSION/config.yaml" config_module/production.yaml

# Restore memory state (optional - preserves user data)
# cp "$BACKUP_DIR/$TARGET_VERSION/memory.csv" memory_module/memory.csv

# Deploy previous version
git checkout $TARGET_VERSION
docker-compose -f production/docker-compose.yml up -d

# Wait for health check
sleep 10
./scripts/health-check.sh

echo "Rollback complete"
```

**Rollback SLA:**
- **Detection:** < 5 minutes (automated health checks)
- **Decision:** < 10 minutes (manual approval or automatic on critical failure)
- **Execution:** < 5 minutes (restore previous Docker image)
- **Total:** < 20 minutes from failure to restored service

**What Gets Rolled Back:**
- ✅ Application code (Docker container image)
- ✅ Configuration files (`config_module/*.yaml`)
- ✅ Dependencies (`requirements.txt` version)
- ❌ Database/Memory state (preserve user data unless corrupted)
- ❌ Infrastructure changes (manual revert required)

**Rollback Decision Matrix:**

| Failure Type | Rollback Strategy | Approval Required |
|--------------|-------------------|-------------------|
| Health check failure after deploy | Automatic rollback | No (happens immediately) |
| Critical bug reported by users | Manual rollback | Yes (1 maintainer) |
| Performance degradation | Manual rollback | Yes (2 maintainers) |
| Security vulnerability in new release | Manual rollback + hotfix | Yes (immediate) |
| Infrastructure issue | Manual investigation first | Yes (DevOps lead) |

---

### 1.6 Branching Strategy

**Recommendation: Simplified GitHub Flow (not GitFlow)**

**Branching Model:**

```
main (protected)
 ├── feature/add-memory-search
 ├── feature/mcp-tool-integration
 ├── fix/state-transition-bug
 └── docs/api-documentation

Tags: v0.1.0, v0.2.0, v0.3.0 (on main)
```

**Why GitHub Flow instead of GitFlow:**
- **Simpler for students:** Less cognitive overhead, easier onboarding
- **Continuous delivery friendly:** No long-lived release branches
- **Small team:** GitFlow's complexity is overkill for <10 active developers
- **Fast iteration:** Features merge quickly, reducing merge conflicts
- **Pre-1.0 software:** Don't need to maintain multiple versions simultaneously

**Branch Naming Conventions:**

| Prefix | Purpose | Example | Merges to |
|--------|---------|---------|-----------|
| `feature/` | New functionality | `feature/add-memory-search` | `main` |
| `fix/` | Bug fixes | `fix/state-transition-bug` | `main` |
| `docs/` | Documentation only | `docs/api-documentation` | `main` |
| `refactor/` | Code refactoring | `refactor/model-interface` | `main` |
| `test/` | Test additions | `test/agent-integration-tests` | `main` |
| `hotfix/` | Urgent production fixes | `hotfix/memory-corruption` | `main` (fast-tracked) |

**Workflow:**

1. **Developer creates feature branch:**
   ```bash
   git checkout -b feature/add-memory-search
   # Work on feature
   git push origin feature/add-memory-search
   ```

2. **Opens Pull Request to `main`:**
   - PR template automatically populated (already exists in `.github/`)
   - Automated CI checks run (to be implemented)
   - 2 peer reviews required

3. **PR Merged to `main`:**
   - Squash merge (cleaner history) or regular merge (preserves commits)
   - Branch automatically deleted
   - Staging environment automatically deploys

4. **Weekly Release:**
   ```bash
   # Maintainer creates release from main
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   # Production deployment triggered with manual approval
   ```

**Branch Protection Rules for `main`:**
```yaml
Branch protection rules:
  ✅ Require pull request before merging
    - Required approvals: 2
    - Dismiss stale reviews when new commits are pushed
    - Require review from code owners
  ✅ Require status checks to pass
    - lint-and-format
    - unit-tests
    - integration-tests
    - security-scan
  ✅ Require branches to be up to date before merging
  ✅ Require conversation resolution before merging
  ✅ Require signed commits (optional, recommended)
  ✅ Do not allow bypassing the above settings
  ❌ Allow force pushes (never enable)
  ❌ Allow deletions (never enable)
```

**Handling Long-Lived Features:**
For features that take >1 week:
- Use **feature flags** instead of long-lived branches
- Merge incomplete features to `main` with flags disabled
- Enable in production when ready

```python
# Example feature flag usage
from config_module.config import config

class Agent:
    def process_query(self, query: str):
        if config.feature_flags.get("memory_search_v2", False):
            return self._memory_search_v2(query)
        else:
            return self._memory_search_v1(query)
```

---

### 1.7 Preview/Ephemeral Environments for PRs

**Recommendation: Not immediately, add in Phase 2 (after 6 months)**

**Current Decision: No ephemeral PR environments**

**Rationale:**
- **GPU resource constraints:** Each SGLANG instance requires significant GPU memory (8GB+)
- **Cost vs. benefit:** Staging environment sufficient for integration testing
- **Team size:** 5-10 developers can coordinate testing in shared staging
- **Complexity:** Ephemeral environments add operational overhead for limited value at this stage

**Alternative: PR Labels for Staging Deployment**

Instead of automatic PR environments, use manual staging deployments for critical PRs:

```yaml
# .github/workflows/deploy-pr-to-staging.yml
name: Deploy PR to Staging
on:
  pull_request:
    types: [labeled]

jobs:
  deploy-pr:
    if: github.event.label.name == 'deploy-to-staging'
    runs-on: ubuntu-latest
    steps:
      - name: Notify team
        run: |
          echo "Deploying PR #${{ github.event.number }} to staging"

      - name: Deploy to staging
        run: ./scripts/deploy-staging.sh ${{ github.event.pull_request.head.sha }}

      - name: Comment on PR
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '🚀 Deployed to staging: http://staging.arkos.internal\nSHA: ${{ github.event.pull_request.head.sha }}'
            })
```

**Future Consideration (Phase 2 - after 6 months):**
When project scales, implement ephemeral environments with:
- **Kubernetes + Knative:** Auto-scaling PR preview deployments
- **Mocked SGLANG:** Use smaller models or mock responses for PR previews
- **Cloud GPU spot instances:** Cost-effective ephemeral GPU compute
- **Namespace isolation:** Each PR gets own K8s namespace

**Trigger for implementing PR environments:**
- Team grows beyond 15 active developers
- Merge conflicts become frequent in staging
- External contributors need isolated testing environments
- Budget allows for additional GPU instances

---

### 1.8 Policy on Broken Builds Blocking Deployments

**Recommendation: Zero-tolerance policy with automated enforcement**

**Policy Statement:**

> **Broken builds MUST NOT be deployed to any environment. All CI checks must pass before merging to `main`, and all staging builds must succeed before production deployment is allowed.**

**Enforcement Mechanisms:**

**1. Pre-Merge Protection (PR Level):**
```yaml
# Required status checks (cannot be bypassed)
Required checks before merge:
  ✅ lint-python (black, flake8, isort)
  ✅ type-check (mypy)
  ✅ unit-tests (pytest)
  ✅ integration-tests (pytest integration/)
  ✅ security-scan (bandit, safety)
  ✅ dependency-check (pip-audit)
  ✅ documentation-build (mkdocs build)

If ANY check fails:
  ❌ Merge button disabled
  ❌ Auto-merge canceled
  ❌ Must fix and re-run checks
```

**2. Post-Merge Protection (Staging Deployment):**
```yaml
# .github/workflows/deploy-staging.yml
jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    steps:
      - name: Run all tests again (defensive)
        run: pytest tests/ --cov=. --cov-report=xml

      - name: Check coverage threshold
        run: |
          coverage report --fail-under=80

      - name: Build Docker image
        run: docker build -t arkos:${{ github.sha }} .

      - name: Only deploy if all steps pass
        if: success()
        run: ./scripts/deploy-staging.sh

      # If deployment fails, notify team
      - name: Notify on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚨 Staging deployment failed for commit ${{ github.sha }}",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Author:* ${{ github.actor }}\n*Branch:* main\n*Action Required:* Fix immediately or revert"
                  }
                }
              ]
            }
```

**3. Production Gate (Release Level):**
```yaml
# .github/workflows/deploy-production.yml
jobs:
  validate-release:
    runs-on: ubuntu-latest
    steps:
      - name: Verify staging is healthy
        run: |
          curl -f http://staging.arkos.internal/health || {
            echo "Staging is broken, cannot deploy to production"
            exit 1
          }

      - name: Run smoke tests on staging
        run: pytest tests/smoke/ --environment=staging

      - name: Check last 10 staging deployments succeeded
        run: |
          # Query GitHub API for last 10 workflow runs
          # Fail if any recent staging deployment failed
          ./scripts/check-staging-health.sh

  deploy-production:
    needs: validate-release
    # Only runs if validate-release succeeds
```

**Breaking the Rules (Emergency Bypass):**

Only allowed in **extreme circumstances:**
- Critical security vulnerability fix
- Production completely down
- Data loss prevention

**Bypass procedure:**
1. Maintainer must document reason in GitHub issue
2. Create hotfix branch with minimal changes
3. Deploy with explicit `--force-deploy` flag
4. Immediately create follow-up PR to fix tests
5. Post-mortem required within 24 hours

```bash
# Emergency deployment script
./scripts/emergency-deploy.sh \
  --version v0.2.1-hotfix \
  --reason "CVE-2024-XXXXX critical security fix" \
  --approver maintainer-1 \
  --skip-tests  # Requires explicit flag
```

**Accountability:**
- All bypasses logged to audit trail
- Monthly review of all emergency deployments
- If >2 emergency deployments per month, process improvements required

**Developer Communication:**

When CI breaks, automated message to PR author:
```
❌ CI Failed for PR #123

Failed checks:
  - unit-tests (3 failures in test_memory.py)
  - lint-python (12 flake8 errors)

Action Required:
1. Review failure logs: [link to workflow run]
2. Fix issues locally: `pytest tests/ && flake8 .`
3. Push fixes to your branch
4. CI will re-run automatically

Need help? Ask in #arkos-dev Slack channel

Merge blocked until all checks pass.
```

---

## 2. Testing & Quality Gates

### 2.1 Required Tests Before Deployment

**Recommendation: Multi-layer testing pyramid with specific gates per environment**

**Testing Pyramid for ARKOS:**

```
                    /\
                   /  \  Manual E2E (optional)
                  /    \
                 /      \  Integration Tests (required for production)
                /        \
               /          \  Unit Tests (required for all merges)
              /            \
             /  Security   \ Linting, Formatting, Type Checks
            /________________\
```

**Test Requirements by Deployment Stage:**

| Test Type | PR Merge to `main` | Staging Deployment | Production Deployment |
|-----------|-------------------|--------------------|-----------------------|
| **Code Linting** (black, flake8) | ✅ Required | ✅ Required | ✅ Required |
| **Type Checking** (mypy) | ✅ Required | ✅ Required | ✅ Required |
| **Unit Tests** | ✅ Required (100% of tests) | ✅ Required | ✅ Required |
| **Integration Tests** | ⚠️ Optional (can skip for minor changes) | ✅ Required | ✅ Required |
| **Security Scans** (SAST) | ✅ Required | ✅ Required | ✅ Required |
| **Dependency Audit** | ✅ Required | ✅ Required | ✅ Required |
| **E2E Tests** | ❌ Not required | ⚠️ Manual verification | ✅ Manual smoke tests |
| **Performance Tests** | ❌ Not required | ⚠️ Weekly (not blocking) | ⚠️ Before major releases |

**Detailed Test Specifications:**

**1. Unit Tests (Required for PR merge)**

**Coverage Requirement:** ≥80% code coverage

**Test Scope:**
```python
# tests/unit/test_agent.py - Example structure
def test_agent_initialization():
    """Test Agent class initializes with correct default state"""
    agent = Agent(agent_id="test-001")
    assert agent.current_state == "greeting"
    assert agent.agent_id == "test-001"

def test_state_transition():
    """Test state transitions follow state graph rules"""
    agent = Agent(agent_id="test-002")
    agent.transition_to("query_understanding")
    assert agent.current_state == "query_understanding"

def test_memory_write_read(tmp_path):
    """Test memory persistence (using temp CSV file)"""
    memory_file = tmp_path / "memory.csv"
    memory = Memory(storage_path=str(memory_file))
    memory.store("user-1", "test_key", "test_value")
    result = memory.retrieve("user-1", "test_key")
    assert result == "test_value"

@pytest.mark.asyncio
async def test_model_call_with_mock():
    """Test model interface without actual LLM call"""
    mock_response = {"choices": [{"message": {"content": "Hello"}}]}
    with patch('model_module.ArkModelNew.generate', return_value=mock_response):
        model = ArkModelNew(base_url="http://mock:8080")
        response = await model.generate(messages=[{"role": "user", "content": "Hi"}])
        assert "Hello" in response
```

**What Must Be Tested:**
- ✅ All public methods in `agent_module/agent.py`
- ✅ State transition logic in `state_module/state_handler.py`
- ✅ Memory CRUD operations in `memory_module/memory.py`
- ✅ Model interface in `model_module/ArkModelNew.py` (with mocked LLM)
- ✅ Tool interface in `tool_module/tool.py`
- ✅ Configuration parsing in `config_module/`
- ❌ Deprecated code in `*/deprecated/` directories (exclude from coverage)

**Test Execution:**
```bash
# Must pass before PR merge
pytest tests/unit/ \
  --cov=agent_module \
  --cov=state_module \
  --cov=memory_module \
  --cov=model_module \
  --cov=tool_module \
  --cov-report=term \
  --cov-report=xml \
  --cov-fail-under=80
```

**2. Integration Tests (Required for staging/production)**

**Coverage Requirement:** All critical user workflows

**Test Scope:**
```python
# tests/integration/test_agent_workflow.py
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_conversation_flow():
    """Test full agent lifecycle with mocked LLM"""
    # Setup
    config = load_config("tests/fixtures/test_config.yaml")
    agent = Agent(agent_id="integration-test", config=config)

    # Simulate full conversation
    with patch_llm_responses("tests/fixtures/mock_responses.json"):
        # Step 1: Greeting
        response1 = await agent.process_input("Hello")
        assert agent.current_state == "greeting"

        # Step 2: Query understanding
        response2 = await agent.process_input("What's the weather?")
        assert agent.current_state == "query_understanding"

        # Step 3: Tool invocation
        assert len(agent.tool_calls) > 0

        # Step 4: Memory persistence
        memory_entry = agent.memory.retrieve_last_conversation()
        assert memory_entry is not None

@pytest.mark.integration
def test_state_machine_all_transitions():
    """Test all possible state transitions defined in state_graph.yaml"""
    state_graph = StateGraph.from_yaml("state_module/state_graph.yaml")
    for state in state_graph.get_all_states():
        for transition in state_graph.get_transitions(state):
            # Verify each transition is reachable
            assert state_graph.can_transition(state, transition)
```

**What Must Be Tested:**
- ✅ Complete conversation flow (greeting → query → tool use → response)
- ✅ State machine traversal (all states reachable)
- ✅ Memory persistence across agent restarts
- ✅ MCP tool integration (with mocked external services)
- ✅ Configuration loading from YAML files
- ✅ Error handling and recovery paths

**Test Execution:**
```bash
# Required before staging deployment
pytest tests/integration/ \
  --maxfail=1 \
  --tb=short \
  --timeout=60  # Integration tests should be fast
```

**3. Security Scans (Required for all deployments)**

**SAST (Static Application Security Testing):**
```yaml
# .github/workflows/security.yml
- name: Run Bandit (Python security linter)
  run: |
    bandit -r agent_module/ memory_module/ model_module/ state_module/ tool_module/ \
      -f json -o bandit-report.json
    # Fail on HIGH severity issues
    bandit -r . -ll

- name: Check dependencies for vulnerabilities
  run: |
    pip-audit --strict  # Fails on any known CVE

- name: Scan for secrets
  uses: trufflesecurity/trufflehog@v3
  with:
    path: ./
    base: main
    head: HEAD
```

**DAST (Dynamic Application Security Testing):**
```yaml
# Future Phase 2: Add OWASP ZAP scanning for API endpoints
# Currently not applicable (no web API in main codebase)
```

**4. E2E Tests (Manual for now, automated in Phase 2)**

**Current Approach: Manual Smoke Tests**

**Pre-Production Checklist:**
```markdown
# Production Deployment Smoke Tests (Manual)

Performed by: [Maintainer Name]
Date: [YYYY-MM-DD]
Version: [v0.x.x]
Environment: Staging (before production deployment)

## Test Cases

- [ ] **Test 1: Agent Initialization**
  - Start agent: `python base_module/main_interface.py`
  - Verify greeting state
  - Verify SGLANG connection (check logs)

- [ ] **Test 2: Simple Conversation**
  - Send query: "Hello, how are you?"
  - Verify response received
  - Verify state transition to appropriate state

- [ ] **Test 3: Memory Persistence**
  - Store a fact: "My name is John"
  - Restart agent
  - Ask: "What's my name?"
  - Verify agent recalls "John"

- [ ] **Test 4: Tool Invocation**
  - Ask question requiring tool use
  - Verify MCP tool called
  - Verify response incorporates tool result

- [ ] **Test 5: Error Handling**
  - Stop SGLANG container
  - Send query
  - Verify graceful error message (no crash)

- [ ] **Test 6: Configuration Loading**
  - Verify correct config file loaded (production.yaml)
  - Verify correct model endpoint
  - Verify logging to correct path

## Results

Pass: [ ] / Fail: [ ]

Notes:
[Any issues encountered]

Approved for production: [ ] Yes [ ] No
```

---

### 2.2 Performance/Load Testing

**Recommendation: Not in initial pipeline, add as periodic manual testing**

**Rationale:**
- **Current project stage:** Early development, performance optimization not critical yet
- **Single-user architecture:** ARKOS is designed for individual users, not high-concurrency
- **GPU bottleneck:** SGLANG inference is the limiting factor, not application code
- **Resource cost:** Load testing requires multiple GPU instances (expensive)

**Alternative: Lightweight Performance Monitoring**

**1. Response Time Tracking (Add to Integration Tests):**
```python
# tests/integration/test_performance.py
import pytest
import time

@pytest.mark.performance
@pytest.mark.asyncio
async def test_agent_response_time():
    """Verify agent responds within acceptable latency"""
    agent = Agent(agent_id="perf-test")

    start = time.time()
    response = await agent.process_input("Hello")
    duration = time.time() - start

    # Assert response within 5 seconds (including LLM call)
    assert duration < 5.0, f"Response took {duration}s, expected <5s"

@pytest.mark.performance
def test_memory_read_performance():
    """Verify memory operations are fast (CSV file performance)"""
    memory = Memory()

    # Write 1000 entries
    for i in range(1000):
        memory.store(f"user-{i}", f"key-{i}", f"value-{i}")

    # Read should be fast even with 1000 entries
    start = time.time()
    result = memory.retrieve("user-500", "key-500")
    duration = time.time() - start

    assert duration < 0.1, f"Memory read took {duration}s, expected <100ms"
```

**2. Benchmarking Script (Run Monthly):**
```python
# scripts/benchmark.py
"""
Monthly performance benchmarking script
Run manually before major releases
"""
import asyncio
from agent_module.agent import Agent
import time
import statistics

async def benchmark_agent_throughput():
    """Measure queries per minute"""
    agent = Agent(agent_id="benchmark")
    queries = ["Test query " + str(i) for i in range(100)]

    start = time.time()
    for query in queries:
        await agent.process_input(query)
    duration = time.time() - start

    qpm = len(queries) / (duration / 60)
    print(f"Throughput: {qpm:.2f} queries/minute")
    print(f"Avg latency: {duration/len(queries):.2f}s per query")

if __name__ == "__main__":
    asyncio.run(benchmark_agent_throughput())
```

**Performance Regression Detection:**
```yaml
# .github/workflows/benchmark.yml (runs weekly)
name: Weekly Performance Benchmark
on:
  schedule:
    - cron: '0 0 * * 0'  # Every Sunday at midnight

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - name: Run benchmark
        run: python scripts/benchmark.py > benchmark-results.txt

      - name: Compare with baseline
        run: |
          # Compare with last week's results
          # Alert if >20% regression
          python scripts/compare-benchmarks.py
```

**When to Add Full Load Testing (Future):**
- When building multi-user web API
- Before handling >100 concurrent users
- When adding real-time features (WebSockets)
- When optimizing for production scale

---

### 2.3 Code Coverage Thresholds

**Recommendation: 80% coverage required for merge, with exceptions**

**Coverage Policy:**

**Required Coverage Levels:**

| Module | Minimum Coverage | Rationale |
|--------|-----------------|-----------|
| `agent_module/` | 85% | Core business logic, high criticality |
| `state_module/` | 90% | State transitions are deterministic, easily testable |
| `memory_module/` | 80% | Data persistence is critical |
| `model_module/` | 70% | LLM integration has external dependencies (harder to test) |
| `tool_module/` | 75% | Tool interfaces have external dependencies |
| `config_module/` | 60% | Configuration parsing (low complexity) |
| `base_module/` | 50% | CLI interface (lower priority for unit tests) |
| **Overall Project** | **80%** | Aggregate across all modules |

**Exemptions (Excluded from Coverage):**
- `*/deprecated/` - Legacy code being phased out
- `*/tests/` - Test code itself
- `*/__init__.py` - Empty init files
- `scripts/` - One-off utility scripts
- `config_module/*.yaml` - Configuration data files

**Implementation:**

**1. pytest configuration (`pytest.ini`):**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --cov=agent_module
    --cov=state_module
    --cov=memory_module
    --cov=model_module
    --cov=tool_module
    --cov=config_module
    --cov-report=term-missing
    --cov-report=html:coverage_html
    --cov-report=xml:coverage.xml
    --cov-fail-under=80
    --strict-markers
    --tb=short

markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (slower, may use mocked services)
    performance: Performance benchmarking tests
```

**2. Coverage reporting in CI:**
```yaml
# .github/workflows/test.yml
- name: Run tests with coverage
  run: |
    pytest tests/ --cov-report=xml --cov-report=term

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    flags: unittests
    fail_ci_if_error: true

- name: Comment coverage on PR
  uses: py-cov-action/python-coverage-comment-action@v3
  with:
    GITHUB_TOKEN: ${{ github.token }}
    MINIMUM_GREEN: 80
    MINIMUM_ORANGE: 70
```

**3. Coverage enforcement:**
```yaml
# .github/workflows/ci.yml
- name: Check coverage thresholds
  run: |
    # Fail if overall coverage < 80%
    coverage report --fail-under=80

    # Check individual module coverage
    coverage report --include="agent_module/*" --fail-under=85
    coverage report --include="state_module/*" --fail-under=90
    coverage report --include="memory_module/*" --fail-under=80
```

**Handling Coverage Exceptions:**

If a PR lowers coverage below threshold:

**Option 1: Add tests to meet threshold (preferred)**
```bash
# Developer adds tests until coverage is met
pytest tests/unit/test_new_feature.py --cov=agent_module --cov-report=term
```

**Option 2: Temporarily allow lower coverage (rare, requires justification)**
```yaml
# PR must include comment explaining why coverage is lower
# Requires explicit approval from maintainer
# Example: External API integration that's hard to mock
# Temporary exemption, must be resolved in follow-up PR
```

**Coverage Dashboard:**
- Integrate with Codecov (free for open source)
- Display coverage badge in README.md
- Track coverage trends over time
- Alert when coverage drops >5% between releases

---

### 2.4 Manual QA Sign-off Before Production

**Recommendation: Yes, but lightweight process**

**QA Process:**

**Who Performs QA:**
- **Primary:** Rotating QA role among MIT student contributors (weekly rotation)
- **Secondary:** DevOps team spot-checks
- **Final Approval:** 1 core maintainer

**QA Checklist (15 minutes max):**

```markdown
# ARKOS Pre-Production QA Checklist

**Release:** v0.x.x
**QA Tester:** [Name]
**Date:** [YYYY-MM-DD]
**Environment:** Staging

## Automated Checks (verify passed)
- [ ] All CI tests passed (check GitHub Actions)
- [ ] Code coverage ≥80%
- [ ] No security vulnerabilities (bandit, pip-audit)
- [ ] Docker image builds successfully

## Manual Testing (perform in staging)

### 1. Basic Functionality (5 min)
- [ ] Agent starts without errors
- [ ] Agent responds to greeting
- [ ] Agent can handle simple query
- [ ] Logs are being written correctly

### 2. State Machine (3 min)
- [ ] Agent transitions through states correctly
- [ ] No unexpected state transitions
- [ ] Error states are handled gracefully

### 3. Memory Persistence (3 min)
- [ ] Agent stores conversation history
- [ ] Restart agent, verify memory persists
- [ ] Memory file is not corrupted

### 4. Configuration (2 min)
- [ ] production.yaml loads correctly
- [ ] Correct model endpoint configured
- [ ] Feature flags set correctly

### 5. Regression Check (2 min)
- [ ] Test previous release's main feature still works
- [ ] No obvious performance degradation

## Release Notes Review
- [ ] Release notes accurately describe changes
- [ ] Breaking changes clearly documented
- [ ] Migration guide provided (if needed)

## Sign-off
- [ ] **QA Passed** - Ready for production
- [ ] **QA Failed** - Issues found (document below)

**Issues Found:**
[List any issues or concerns]

**QA Signature:** [Name]
**Maintainer Approval:** [Name]
```

**QA Timing:**
- **Frequency:** Every production release (weekly)
- **Duration:** 15 minutes max
- **Blocking:** Yes - production deployment waits for QA approval

**When QA Can Be Skipped:**
- Hotfix deployments (critical security fixes)
- Documentation-only releases
- Configuration changes (no code changes)

**QA Automation (Phase 2):**
Eventually automate portions of this checklist:
- Automated smoke tests in staging
- Visual regression testing (for UI, when added)
- Automated performance comparisons

---

## 3. Collaboration & Ownership

### 3.1 CI/CD Pipeline Ownership

**Recommendation: Shared ownership with clear primary responsible party**

**Ownership Model:**

```
┌─────────────────────────────────────────────────────────┐
│  PRIMARY OWNER: DevOps Team (You)                        │
│  - GitHub Actions workflow maintenance                   │
│  - Infrastructure provisioning (GPU servers)             │
│  - Deployment automation scripts                         │
│  - Monitoring and alerting setup                         │
│  - Incident response for pipeline failures               │
│  - Quarterly pipeline improvements                       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  SECONDARY OWNERS: Core Maintainers (2-3 people)         │
│  - Review workflow changes in PRs                        │
│  - Approve production deployments                        │
│  - Escalation point for deployment decisions             │
│  - Define quality gates and coverage thresholds          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  CONTRIBUTORS: MIT Students                              │
│  - Fix their own CI failures in PRs                      │
│  - Add tests for new features                            │
│  - Suggest pipeline improvements                         │
│  - Rotate QA testing responsibilities                    │
└─────────────────────────────────────────────────────────┘
```

**Responsibilities Matrix:**

| Task | DevOps Team | Core Maintainers | Contributors |
|------|-------------|------------------|--------------|
| **Create new GitHub Actions workflows** | Primary | Review | Suggest |
| **Fix broken CI pipeline** | Primary | Escalate to DevOps | Report issue |
| **Update dependency versions** | Review | Primary | Submit PR |
| **Approve production deployments** | Execute | Approve (2 required) | Not authorized |
| **Configure monitoring/alerting** | Primary | Define requirements | - |
| **Add tests for new features** | Review | Review | Primary |
| **Fix failing tests in PRs** | Help debug | Help debug | Primary |
| **Infrastructure provisioning** | Primary | Approve budget | - |
| **Security vulnerability remediation** | Coordinate | Approve fixes | Implement fixes |
| **Pipeline performance optimization** | Primary | Review | Suggest |

**Communication Channels:**

**1. GitHub Issues for Pipeline Problems:**
```markdown
# Issue Template: .github/ISSUE_TEMPLATE/ci-pipeline-issue.md

---
name: CI/CD Pipeline Issue
about: Report a problem with the CI/CD pipeline
title: '[CI] '
labels: 'ci-cd, devops'
assignees: 'devops-team'
---

## Issue Description
[Clear description of the pipeline problem]

## Affected Workflow
- [ ] CI (pull request checks)
- [ ] Staging deployment
- [ ] Production deployment
- [ ] Security scanning
- [ ] Other: __________

## Failure Logs
[Paste relevant logs or link to workflow run]

## Impact
- [ ] Blocking all PRs
- [ ] Blocking staging deployment
- [ ] Blocking production deployment
- [ ] No immediate impact

## Steps Attempted
[What you've tried to fix it]
```

**2. Slack Channels (if applicable):**
- `#arkos-dev` - General development discussions
- `#arkos-deployments` - Deployment notifications and coordination
- `#arkos-alerts` - Automated alerts from CI/CD failures

**3. Monthly CI/CD Review Meeting:**
- **Attendees:** DevOps team + core maintainers
- **Agenda:**
  - Review pipeline reliability metrics (success rate, MTTR)
  - Discuss pain points reported by contributors
  - Plan improvements for next month
  - Review security scan results

**Escalation Path:**

```
Level 1: Contributor encounters CI failure
   ↓ (tries to fix for 30 min)
Level 2: Ask in #arkos-dev Slack channel
   ↓ (if not resolved in 1 hour)
Level 3: Create GitHub issue, tag @devops-team
   ↓ (if blocking production)
Level 4: Direct message DevOps team lead + core maintainer
```

---

### 3.2 Access to Logs and Debugging Tools

**Recommendation: Tiered access with self-service for common cases**

**Log Access Levels:**

**Level 1: PR CI Logs (Public - All Developers)**

**Access Method:**
- GitHub Actions workflow logs (publicly visible for open-source repo)
- Click "Details" next to failed check in PR

**What's Available:**
```
✅ Unit test failures with stack traces
✅ Linting errors with line numbers
✅ Build errors
✅ Dependency installation issues
✅ Type checking errors
```

**Self-Service Debugging:**
```bash
# Developers can reproduce locally
# Copy commands from failed workflow

# Example from .github/workflows/ci.yml
- name: Run tests
  run: pytest tests/ --cov=. --cov-report=term

# Developer runs same command locally
pytest tests/ --cov=. --cov-report=term
```

**Level 2: Staging Deployment Logs (Authenticated - Contributors)**

**Access Method:**
- SSH access to staging server (request via GitHub issue)
- Grafana dashboard (read-only access)

**What's Available:**
```
✅ Application logs: /var/log/arkos/staging.log
✅ SGLANG logs: docker logs sglang-staging
✅ System metrics (CPU, memory, GPU usage)
✅ Deployment history (last 30 days)
```

**How to Request Access:**
```markdown
# Create issue: "Request Staging Log Access"
Reason: [Debugging feature X]
Duration needed: [7 days / 30 days / permanent]
Approval: [Maintainer approves]
Provisioning: [DevOps team creates SSH key, adds to authorized_keys]
```

**Level 3: Production Logs (Restricted - Maintainers + DevOps)**

**Access Method:**
- SSH access with 2FA required
- VPN connection to production network
- Audit logging enabled (all access recorded)

**What's Available:**
```
✅ Production application logs
✅ Production SGLANG logs
✅ System metrics and dashboards
✅ Deployment history (full retention)
✅ User data (with strict privacy controls)
```

**Access Control:**
```bash
# Only specific SSH keys allowed
# /etc/ssh/authorized_keys on production server

# devops-team-lead
ssh-rsa AAAAB3... devops-lead@mit.edu

# maintainer-1
ssh-rsa AAAAB3... maintainer1@mit.edu

# All access logged
# auditd configuration monitors SSH sessions
```

**Debugging Tools Provided:**

**1. Debugging Scripts (in repo):**
```bash
# scripts/debug-ci.sh
#!/bin/bash
# Reproduce CI environment locally
docker run -it --rm \
  -v $(pwd):/workspace \
  python:3.11-slim \
  /bin/bash -c "
    cd /workspace
    pip install -r requirements.txt -r requirements-dev.txt
    pytest tests/
    flake8 .
    mypy .
  "

# scripts/tail-logs.sh
#!/bin/bash
# Tail staging logs (requires SSH access)
ssh staging.arkos.internal "tail -f /var/log/arkos/staging.log"

# scripts/check-deployment-status.sh
#!/bin/bash
# Check if staging/production is healthy
curl -f http://staging.arkos.internal/health && echo "Staging OK"
curl -f http://production.arkos.internal/health && echo "Production OK"
```

**2. Grafana Dashboards (Future Phase 2):**
```
Dashboard: ARKOS Application Metrics
- Request rate (queries per minute)
- Response latency (p50, p95, p99)
- Error rate
- SGLANG GPU utilization
- Memory usage over time

Dashboard: CI/CD Pipeline Metrics
- Build success rate (last 30 days)
- Test execution time trends
- Deployment frequency
- Mean time to recovery (MTTR)
```

**3. Log Aggregation (Future Phase 2):**
```yaml
# Use Loki or Elasticsearch for log search
# Example query:

# Find all errors in last 24 hours
level:ERROR timestamp:>now-24h

# Find failed state transitions
message:"state transition failed" agent_id:*

# Find slow LLM calls
duration_ms:>5000 module:model_module
```

**Documentation for Debugging:**

Create `docs/debugging-guide.md`:
```markdown
# ARKOS Debugging Guide

## PR CI Failures

### Unit Test Failures
1. Click "Details" next to failed check
2. Expand failed test output
3. Copy stack trace
4. Reproduce locally: `pytest tests/unit/test_file.py::test_name -v`

### Linting Failures
1. View flake8 errors in CI log
2. Run locally: `flake8 .`
3. Auto-fix formatting: `black . && isort .`

### Type Checking Failures
1. View mypy errors in CI log
2. Run locally: `mypy .`
3. Add type hints to fix errors

## Staging Deployment Issues

### How to Access Staging Logs
1. Request SSH access via GitHub issue
2. SSH into staging: `ssh staging.arkos.internal`
3. Tail logs: `tail -f /var/log/arkos/staging.log`

### Common Issues
- SGLANG not responding: `docker restart sglang-staging`
- Memory file corrupted: Restore from backup
- Config not loading: Check YAML syntax

## Production Issues (Maintainers Only)

### Emergency Access
1. Connect to MIT VPN
2. SSH with 2FA: `ssh production.arkos.internal`
3. Check service status: `systemctl status arkos`

### Rollback Procedure
See ROLLBACK.md
```

---

### 3.3 Process When Deployment Fails

**Recommendation: Clear runbook with automated notifications**

**Deployment Failure Response Process:**

**Phase 1: Detection (Automated)**

**Failure Triggers:**
```yaml
# .github/workflows/deploy-staging.yml
jobs:
  deploy:
    steps:
      # ... deployment steps ...

      - name: Notify on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          channel-id: 'arkos-deployments'
          payload: |
            {
              "text": "🚨 STAGING DEPLOYMENT FAILED",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Staging Deployment Failed*\n*Commit:* ${{ github.sha }}\n*Author:* ${{ github.actor }}\n*Branch:* ${{ github.ref }}\n*Workflow:* <${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View Logs>"
                  }
                },
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Action Required:* Review logs and fix immediately or revert commit"
                  }
                }
              ]
            }
```

**Email Notification (for production failures):**
```yaml
- name: Email on production failure
  if: failure()
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.mit.edu
    server_port: 587
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: "🚨 PRODUCTION DEPLOYMENT FAILED - v${{ github.ref_name }}"
    to: devops-team@mit.edu,maintainers@mit.edu
    from: arkos-ci@mit.edu
    body: |
      Production deployment has failed for version ${{ github.ref_name }}.

      Commit: ${{ github.sha }}
      Author: ${{ github.actor }}
      Workflow: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}

      IMMEDIATE ACTION REQUIRED: Review logs and initiate rollback if necessary.
```

**Phase 2: Triage (5 minutes)**

**Responsible Party:**
- **Staging failure:** Original commit author
- **Production failure:** On-call maintainer (weekly rotation)

**Triage Checklist:**
```markdown
## Deployment Failure Triage

**Time Detected:** [HH:MM]
**Environment:** [Staging / Production]
**Severity:**
  - [ ] Critical (Production down, users impacted)
  - [ ] High (Staging down, blocking development)
  - [ ] Medium (Partial failure, degraded service)
  - [ ] Low (Non-blocking, can fix later)

## Quick Assessment (5 min max)

1. **What failed?**
   - [ ] Build step (Docker image creation)
   - [ ] Deployment step (pushing to server)
   - [ ] Health check (post-deployment verification)
   - [ ] Other: __________

2. **Is the service still running?**
   - [ ] Yes, previous version still serving traffic
   - [ ] No, service is down

3. **Is this a known issue?**
   - [ ] Check recent similar failures in GitHub issues
   - [ ] Check if others are experiencing same problem

4. **Can it be fixed quickly (<15 min)?**
   - [ ] Yes - Obvious fix (typo, config error)
   - [ ] No - Requires investigation

## Decision

- [ ] **FIX FORWARD:** Quick fix, push new commit
- [ ] **ROLLBACK:** Revert to previous version
- [ ] **INVESTIGATE:** Need more time to debug (rollback first if production)
```

**Phase 3: Resolution**

**Option A: Fix Forward (Preferred for staging, minor issues)**

```bash
# Author makes quick fix
git checkout -b fix/deployment-failure
# Fix the issue
git commit -m "fix: resolve deployment failure in X"
git push origin fix/deployment-failure

# Create PR (fast-tracked, skip some checks if needed)
# Mark as "hotfix" to expedite reviews
```

**Option B: Rollback (Required for production, critical issues)**

```bash
# Maintainer triggers rollback workflow
gh workflow run rollback-production.yml \
  -f target_version=v0.1.9 \
  -f reason="Deployment failed health checks"

# Workflow automatically:
# 1. Deploys previous version
# 2. Verifies health checks pass
# 3. Notifies team
```

**Option C: Emergency Manual Intervention**

```bash
# SSH into affected server
ssh staging.arkos.internal

# Check service status
systemctl status arkos
docker ps
docker logs arkos-agent

# Manual rollback if needed
cd /opt/arkos
git checkout v0.1.9
docker-compose down
docker-compose up -d

# Verify health
curl http://localhost:8080/health
```

**Phase 4: Post-Mortem (Within 48 hours)**

**For Production Failures Only:**

```markdown
# Post-Mortem Template: .github/ISSUE_TEMPLATE/postmortem.md

---
name: Deployment Post-Mortem
about: Document lessons learned from production deployment failure
title: '[POST-MORTEM] '
labels: 'post-mortem, production'
---

## Incident Summary
**Date:** YYYY-MM-DD
**Duration:** [Time from failure to resolution]
**Severity:** [Critical / High / Medium]
**Affected Environment:** Production

## Timeline
- [HH:MM] Deployment initiated
- [HH:MM] Failure detected (automated alert)
- [HH:MM] Team notified
- [HH:MM] Triage completed
- [HH:MM] Decision: [Rollback / Fix forward]
- [HH:MM] Service restored

## Root Cause
[What caused the failure?]

## Impact
- Downtime: [X minutes]
- Users affected: [Number or percentage]
- Data loss: [Yes/No, describe if yes]

## What Went Well
- [Positive aspects of response]

## What Went Wrong
- [Issues in response or detection]

## Action Items
- [ ] [Specific improvement with owner and deadline]
- [ ] [Another improvement]

## Prevention
- [ ] Add automated test to catch this issue
- [ ] Update deployment checklist
- [ ] Improve monitoring/alerting

**Prepared by:** [Name]
**Reviewed by:** [Maintainer]
```

---

### 3.4 Notification Strategy

**Recommendation: Multi-channel with severity-based routing**

**Notification Matrix:**

| Event | Severity | Slack | Email | GitHub | Required Response Time |
|-------|----------|-------|-------|--------|----------------------|
| **PR CI Failed** | Low | ❌ | ❌ | ✅ (PR comment) | Fix before merge |
| **Staging Deployment Failed** | Medium | ✅ #arkos-deployments | ❌ | ✅ (Issue created) | Fix within 2 hours |
| **Production Deployment Failed** | Critical | ✅ @channel in #arkos-alerts | ✅ Maintainers + DevOps | ✅ (Issue + email) | Fix within 15 min |
| **Production Health Check Failed** | Critical | ✅ @channel | ✅ Immediate | ✅ Auto-rollback | Fix within 5 min |
| **Security Vulnerability Detected** | High | ✅ #arkos-security | ✅ Maintainers | ✅ (Private issue) | Fix within 24 hours |
| **Weekly Build Success** | Info | ✅ (summary) | ❌ | ❌ | N/A |
| **Deployment Success (Production)** | Info | ✅ #arkos-deployments | ❌ | ✅ (Release notes) | N/A |

**Notification Examples:**

**1. PR CI Failure (GitHub Comment):**
```markdown
## ❌ CI Checks Failed

Your pull request has failed the following checks:

### Failed Checks
- **unit-tests**: 3 tests failed in `test_memory.py`
- **lint-python**: 12 flake8 errors

### Next Steps
1. Review the [workflow logs](link to run) for details
2. Fix issues locally:
   ```bash
   pytest tests/unit/test_memory.py
   flake8 .
   ```
3. Push fixes to your branch - CI will re-run automatically

### Need Help?
- Check the [debugging guide](link)
- Ask in #arkos-dev Slack channel
- Tag @maintainers in this PR

---
*This comment was automatically generated by GitHub Actions*
```

**2. Staging Deployment Failure (Slack):**
```
🚨 Staging Deployment Failed

Commit: abc123d
Author: @john-doe
Branch: main
Time: 2025-11-01 14:32 UTC

Failed Step: health-check
Error: SGLANG connection timeout

Action Required:
@john-doe please investigate and fix within 2 hours

View Logs: [Link to GitHub Actions]
Related PR: #456
```

**3. Production Deployment Success (Slack + GitHub Release):**
```
✅ Production Deployment Successful

Version: v0.2.0
Deployed: 2025-11-01 15:00 UTC
Deployed by: @maintainer-1

Changes in this release:
• Added memory search functionality
• Fixed state transition bug
• Improved LLM response parsing

Release Notes: [Link to GitHub Release]
Rollback Plan: [Link to runbook]

All health checks passed ✓
```

**4. Security Vulnerability Alert (Email + Private Issue):**
```
Subject: 🔒 Security Vulnerability Detected in ARKOS Dependencies

A security vulnerability has been detected:

Package: requests
Current Version: 2.32.0
Vulnerable To: CVE-2024-XXXXX (HIGH severity)
Recommended Action: Upgrade to requests>=2.32.3

A private security issue has been created: GHSA-xxxx-xxxx-xxxx

Please review and patch within 24 hours.

Automated Fix Available: Run `pip-audit --fix`
```

**Implementation:**

**GitHub Actions Notification Steps:**
```yaml
# Reusable notification workflow
# .github/workflows/notify.yml
name: Send Notification

on:
  workflow_call:
    inputs:
      message:
        required: true
        type: string
      severity:
        required: true
        type: string
      channel:
        required: false
        type: string
        default: 'arkos-deployments'

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Send Slack notification
        if: inputs.severity != 'low'
        uses: slackapi/slack-github-action@v1
        with:
          channel-id: ${{ inputs.channel }}
          payload: ${{ inputs.message }}

      - name: Send email if critical
        if: inputs.severity == 'critical'
        uses: dawidd6/action-send-mail@v3
        with:
          # email configuration
          body: ${{ inputs.message }}

      - name: Create GitHub issue if failed
        if: inputs.severity == 'critical' || inputs.severity == 'high'
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `[AUTO] Deployment failure detected`,
              body: ${{ inputs.message }},
              labels: ['deployment-failure', 'automated']
            })
```

---

### 3.5 Developer Self-Service Deployments

**Recommendation: Yes for staging, No for production**

**Self-Service Model:**

**Staging Deployments:**

✅ **Fully Self-Service**

```yaml
# Automatic staging deployment on PR merge
# .github/workflows/deploy-staging.yml
on:
  push:
    branches: [main]

# No approval required
# Developers can merge their own PRs (after reviews)
# Staging automatically updates
```

**Benefits:**
- Fast iteration cycles
- Developers get immediate feedback
- Reduces bottlenecks
- Encourages experimentation

**Production Deployments:**

❌ **Not Self-Service (Requires Approval)**

```yaml
# .github/workflows/deploy-production.yml
on:
  release:
    types: [published]

jobs:
  deploy:
    environment: production  # Requires manual approval
    steps:
      # ... deployment steps
```

**Why Not Self-Service for Production:**
- Risk mitigation (prevent accidental deployments)
- Ensures release notes are complete
- Allows final review of changes
- Maintains accountability for production changes
- Students are learning - supervision is appropriate

**Hybrid Approach: Staging Preview for PRs**

Developers CAN trigger staging deployment of their PR:

```yaml
# .github/workflows/deploy-pr-to-staging.yml
on:
  issue_comment:
    types: [created]

jobs:
  deploy-pr-preview:
    if: |
      github.event.issue.pull_request &&
      contains(github.event.comment.body, '/deploy-staging')
    runs-on: ubuntu-latest
    steps:
      - name: Comment acknowledgment
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '🚀 Deploying PR to staging environment...'
            })

      - name: Deploy PR to staging
        run: ./scripts/deploy-staging.sh ${{ github.event.pull_request.head.sha }}

      - name: Comment with result
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '✅ PR deployed to staging: http://staging.arkos.internal\n\nThis deployment will be replaced by the next `/deploy-staging` command or merge to main.'
            })
```

**Usage:**
```markdown
# Developer comments on their PR:
/deploy-staging

# Bot responds:
✅ PR deployed to staging: http://staging.arkos.internal
```

---

### 3.6 Communication Protocol During Deployments

**Recommendation: Structured communication with pre-deployment announcements**

**Deployment Communication Workflow:**

**Staging Deployments (Automated, Low-Ceremony)**

```
[10 minutes before] ❌ No announcement (happens frequently)
[During deployment] Slack notification: "🚀 Staging deploying..."
[After deployment]  Slack notification: "✅ Staging deployed"
```

**Production Deployments (Manual, High-Ceremony)**

**Timeline:**

```
T-24 hours: Release candidate announced
   ↓
T-2 hours: Pre-deployment checklist started
   ↓
T-30 min: Deployment window announced (no merges to main)
   ↓
T-0: Deployment begins
   ↓
T+5 min: Health checks pass, deployment complete
   ↓
T+15 min: Post-deployment verification
   ↓
T+1 hour: All-clear, normal operations resume
```

**Detailed Communication:**

**T-24 Hours: Release Announcement**
```markdown
# Posted in #arkos-deployments

📦 **Release v0.2.0 Scheduled**

**Deployment Time:** Friday, Nov 1, 2025 at 15:00 UTC (10am EST)
**Duration:** ~30 minutes
**Impact:** No expected downtime

**What's New:**
• Added memory search functionality (#123)
• Fixed state transition bug (#145)
• Improved LLM response parsing (#156)

**Release Notes:** [Link to GitHub Release]

**Pre-Deployment Checklist:**
- [ ] QA testing complete (@qa-tester-name)
- [ ] Documentation updated (@docs-team)
- [ ] Rollback plan verified (@devops-team)
- [ ] Stakeholders notified

**Questions?** Reply in thread or DM @maintainer-lead
```

**T-2 Hours: Pre-Deployment Checklist**
```markdown
# Slack thread update

🔍 **Pre-Deployment Checklist Started**

- [x] All CI checks passed
- [x] Staging deployment successful (deployed 2 days ago)
- [x] QA sign-off received
- [x] Rollback plan confirmed (revert to v0.1.9)
- [ ] Final smoke tests running...

**Next Step:** Deployment window begins at 14:30 UTC (30 min before deployment)
```

**T-30 Min: Deployment Window**
```markdown
# Posted in #arkos-deployments

🚨 **Deployment Window OPEN**

**Code Freeze:** No merges to `main` until deployment complete
**Duration:** 30 minutes (14:30 - 15:00 UTC)

**What's Happening:**
1. Final production backup
2. Deploy v0.2.0 to production
3. Health checks
4. Post-deployment verification

**On-Call:** @devops-lead, @maintainer-1

**Stay Tuned:** Will post updates here
```

**T-0: Deployment Begins**
```markdown
🚀 **Production Deployment IN PROGRESS**

Version: v0.2.0
Started: 15:00 UTC
Estimated completion: 15:05 UTC

[Progress Bar] ▓▓▓▓▓▓░░░░ 60%

Current Step: Running database migrations...
```

**T+5 Min: Deployment Complete**
```markdown
✅ **Production Deployment SUCCESSFUL**

Version: v0.2.0
Completed: 15:05 UTC
Duration: 5 minutes

**Health Checks:**
✅ Application responding
✅ SGLANG connectivity confirmed
✅ Memory operations working
✅ Smoke tests passed

**Post-Deployment Verification:** In progress (15 min)

**Code Freeze:** Still in effect until all-clear
```

**T+15 Min: Post-Deployment Verification**
```markdown
✅ **Post-Deployment Verification COMPLETE**

All systems operational:
✅ 100 test queries processed successfully
✅ Response times within expected range (<3s avg)
✅ No error spikes in logs
✅ Memory persistence confirmed

**Code Freeze LIFTED**

Normal operations resume. Thank you for your patience!

**Release Notes:** [Link]
**Monitoring Dashboard:** [Link]
```

**Communication for Issues:**

**If Deployment Fails:**
```markdown
🚨 **Production Deployment FAILED**

Version: v0.2.0
Failed at: 15:03 UTC
Failed step: Health checks

**Status:** Automatic rollback initiated
**Previous version (v0.1.9):** Being restored

**Impact:** ~5 minutes downtime expected

**On-Call Team:** Investigating root cause
**Updates:** Will post every 5 minutes

**DO NOT MERGE** any PRs until all-clear given
```

**Rollback Communication:**
```markdown
🔄 **Rollback COMPLETE**

Previous version: v0.1.9
Restored at: 15:10 UTC
Service status: ✅ Operational

**Root Cause:** Health checks failed (SGLANG connection timeout)
**Next Steps:**
1. Post-mortem investigation
2. Fix identified issue
3. Reschedule deployment

**Code Freeze LIFTED**

**Apologies for disruption.** We'll share findings in post-mortem.
```

---

### 3.7 Hotfix vs Regular Release Process

**Recommendation: Separate fast-track hotfix process with clear criteria**

**Process Comparison:**

| Aspect | Regular Release | Hotfix Release |
|--------|----------------|----------------|
| **Trigger** | Weekly schedule | Critical bug/security issue |
| **Timeline** | 7 days (full week) | <4 hours |
| **Testing** | Full test suite + QA | Minimal tests (affected area only) |
| **Reviews** | 2 peer reviews required | 1 maintainer review required |
| **Approval** | 2 maintainers | 1 maintainer (can be author if maintainer) |
| **Deployment** | Manual approval gate | Fast-tracked (can bypass some checks) |
| **Communication** | 24-hour notice | Immediate notification |
| **Branch** | `main` → release tag | `hotfix/fix-name` → `main` → release tag |

**Hotfix Criteria (Must Meet ONE of These):**

1. **Security Vulnerability**
   - CVE with HIGH or CRITICAL severity
   - Exploitable vulnerability reported by user
   - Credentials or secrets exposed

2. **Production Outage**
   - Service completely down
   - Critical feature broken (e.g., agent can't process queries)
   - Data corruption or loss

3. **Severe Bug**
   - Affects all users
   - No workaround available
   - User data at risk

**NOT a Hotfix (Use Regular Release):**
- Minor bugs with workarounds
- Feature requests
- Performance optimizations (unless catastrophic)
- Documentation fixes
- Cosmetic issues

---

**Hotfix Workflow:**

**Step 1: Hotfix Decision (5 minutes)**

```markdown
# Create GitHub issue immediately

---
name: Hotfix Request
title: '[HOTFIX] Critical bug: Agent crashes on empty query'
labels: 'hotfix, critical, production'
assignees: '@devops-team, @maintainers'
---

## Hotfix Justification
**Severity:** [Security / Outage / Severe Bug]
**Impact:** Production completely down, all users affected
**Users Affected:** 100% of production users
**Workaround Available:** No

## Root Cause
Agent crashes when receiving empty query string due to missing null check

## Proposed Fix
Add null check in `agent.py:process_input()` method

## Estimated Fix Time
30 minutes (simple one-line fix)

## Hotfix Approval
- [ ] Approved by: @maintainer-1
- [ ] DevOps notified: @devops-team
```

**Step 2: Create Hotfix Branch (5 minutes)**

```bash
# Branch from latest production release tag
git checkout v0.2.0
git checkout -b hotfix/agent-empty-query-crash

# Make minimal fix
# Edit agent.py
git add agent_module/agent.py
git commit -m "hotfix: add null check for empty query input

Fixes critical production bug where agent crashes on empty query.

Root cause: Missing null check in process_input() method.
Fix: Added early return for empty/null queries.

Resolves #789"

# Push hotfix branch
git push origin hotfix/agent-empty-query-crash
```

**Step 3: Fast-Tracked PR (15 minutes)**

```yaml
# .github/workflows/hotfix-ci.yml
# Lighter CI checks for hotfixes
name: Hotfix CI
on:
  pull_request:
    branches: [main]
    types: [opened, synchronize]
    paths:
      - 'hotfix/**'

jobs:
  fast-checks:
    runs-on: ubuntu-latest
    steps:
      # Only run tests for affected module
      - name: Run unit tests (affected module only)
        run: pytest tests/unit/test_agent.py

      # Skip integration tests for speed
      # Skip some linting checks

      - name: Security scan (critical issues only)
        run: bandit -r agent_module/ -lll  # Only HIGH severity
```

**PR Requirements for Hotfix:**
- ✅ 1 maintainer approval (instead of 2)
- ✅ Unit tests pass for affected module
- ✅ Security scan passes
- ❌ No need for full integration test suite
- ❌ No need for QA sign-off (will verify in production monitoring)

**Step 4: Merge & Tag (5 minutes)**

```bash
# Maintainer merges PR
gh pr merge hotfix/agent-empty-query-crash --squash

# Immediately create patch release tag
git checkout main
git pull
git tag -a v0.2.1 -m "Hotfix v0.2.1: Fix agent crash on empty query

Emergency hotfix for critical production bug.

Fixes: #789
Released: 2025-11-01 16:45 UTC"

git push origin v0.2.1
```

**Step 5: Deploy to Production (15 minutes)**

```yaml
# .github/workflows/deploy-production-hotfix.yml
# Separate workflow for hotfix deployments
name: Deploy Hotfix to Production

on:
  push:
    tags:
      - 'v*.*.1'  # Patch version triggers hotfix deployment
      - 'v*.*.2'
      - 'v*.*.3'
      # etc.

jobs:
  deploy-hotfix:
    runs-on: ubuntu-latest
    environment: production-hotfix  # Separate environment with faster approval
    steps:
      - name: Require 1 approval (instead of 2)
        # ... deployment steps same as regular production deploy

      - name: Deploy immediately after approval
        run: ./scripts/deploy-production.sh ${{ github.ref_name }}

      - name: Monitor closely for 30 minutes
        run: ./scripts/monitor-deployment.sh --duration 30m
```

**Step 6: Communication**

```markdown
# Slack: #arkos-alerts

🚨 **HOTFIX DEPLOYED TO PRODUCTION**

**Version:** v0.2.1
**Deployed:** 2025-11-01 16:45 UTC
**Reason:** Critical bug - agent crash on empty query

**Fix Applied:**
Added null check in agent.process_input() to prevent crashes

**Impact:**
• Downtime: ~10 minutes during deployment
• Users affected: All production users
• Resolution: Service restored and stable

**Monitoring:**
Enhanced monitoring active for next 30 minutes
On-call team standing by

**Post-Mortem:**
Will be published within 48 hours (#790)

**Questions?** Reply in thread
```

**Step 7: Post-Deployment Monitoring (30 minutes)**

```bash
# scripts/monitor-deployment.sh
#!/bin/bash
DURATION=${1:-30m}

echo "Monitoring production for $DURATION"
echo "Checking every 1 minute..."

# Watch error logs
watch -n 60 'curl -s http://production.arkos.internal/metrics | grep error_count'

# Alert if error rate spikes
# Alert if response time degrades
# Alert if any crashes detected
```

**Step 8: Post-Mortem (Within 48 hours)**

Required for all hotfixes - see Post-Mortem template in section 3.3.

---

**Hotfix vs Regular Release Decision Tree:**

```
Is production broken?
├─ YES ──> Is it CRITICAL (outage/security/data loss)?
│          ├─ YES ──> HOTFIX PROCESS
│          └─ NO ──> Can it wait until next weekly release?
│                     ├─ YES ──> REGULAR RELEASE
│                     └─ NO ──> HOTFIX PROCESS (with justification)
└─ NO ──> REGULAR RELEASE
```

---

## 4. Developer Experience

### 4.1 Self-Service CI/CD for Developers

**Recommendation: Yes, with comprehensive documentation and tooling**

**Self-Service Capabilities:**

**What Developers CAN Do Themselves:**

✅ **1. Run Full CI Pipeline Locally**

```bash
# scripts/run-ci-locally.sh
#!/bin/bash
set -e

echo "Running full CI pipeline locally..."

echo "1/6 Code formatting check..."
black --check .
isort --check .

echo "2/6 Linting..."
flake8 .

echo "3/6 Type checking..."
mypy .

echo "4/6 Security scan..."
bandit -r . -ll

echo "5/6 Unit tests..."
pytest tests/unit/ --cov=. --cov-report=term

echo "6/6 Integration tests..."
pytest tests/integration/

echo "✅ All CI checks passed! Your PR is ready."
```

**Usage:**
```bash
# Before creating PR, developer runs:
./scripts/run-ci-locally.sh

# Catches issues before pushing to GitHub
# Saves CI minutes and embarrassment
```

✅ **2. Deploy Their PR to Staging**

```bash
# Comment on PR:
/deploy-staging

# Or manually trigger workflow:
gh workflow run deploy-pr-to-staging.yml \
  -f pr_number=123
```

✅ **3. View Deployment Status**

```bash
# scripts/check-deployment.sh
#!/bin/bash
ENVIRONMENT=${1:-staging}

echo "Checking $ENVIRONMENT deployment status..."

# Check service health
curl -f http://$ENVIRONMENT.arkos.internal/health

# Check recent deployments
gh run list --workflow=deploy-$ENVIRONMENT.yml --limit 5

# Check current version
curl -s http://$ENVIRONMENT.arkos.internal/version
```

✅ **4. Rollback Staging (Their Own PRs Only)**

```bash
# If developer's PR breaks staging, they can rollback
gh workflow run rollback-staging.yml \
  -f target_commit=$(git rev-parse HEAD~1)
```

✅ **5. View Logs**

```bash
# scripts/view-logs.sh
#!/bin/bash
ENVIRONMENT=${1:-staging}
LINES=${2:-100}

if [ "$ENVIRONMENT" = "staging" ]; then
  # Anyone can view staging logs
  ssh staging.arkos.internal "tail -n $LINES /var/log/arkos/staging.log"
elif [ "$ENVIRONMENT" = "production" ]; then
  echo "Production logs require maintainer access. Request access via GitHub issue."
  exit 1
else
  echo "Unknown environment: $ENVIRONMENT"
  exit 1
fi
```

**What Developers CANNOT Do:**

❌ **1. Deploy to Production**
- Requires maintainer approval
- Must go through release process

❌ **2. Modify GitHub Actions Workflows**
- Requires review from DevOps team
- Can propose changes via PR, but cannot merge without approval

❌ **3. Bypass CI Checks**
- No force-merge capability
- All checks must pass

❌ **4. Access Production Logs/Data**
- Requires explicit maintainer/DevOps permission
- Privacy and security concerns

**Self-Service Tools Provided:**

**1. CLI Tool for Common Tasks**

```bash
# arkos-cli (to be created)
# Installation: pip install arkos-cli

# Check CI status
arkos-cli check-ci --pr 123

# Deploy to staging
arkos-cli deploy --env staging --pr 123

# View logs
arkos-cli logs --env staging --follow

# Run tests
arkos-cli test --local --verbose

# Validate config
arkos-cli validate-config config_module/staging.yaml
```

**2. Pre-commit Hooks (Automatic)**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203,W503']

  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest-fast
        entry: pytest tests/unit/ -x --tb=short
        language: system
        pass_filenames: false
        stages: [commit]
```

**Installation:**
```bash
# Developers install once
pip install pre-commit
pre-commit install

# Now runs automatically on every commit
git commit -m "add feature"
# → black, isort, flake8, pytest run automatically
# → Commit only succeeds if all pass
```

**3. VS Code Integration (`.vscode/settings.json`)**

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.flake8Args": ["--max-line-length=100"],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length=100"],
  "editor.formatOnSave": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "files.watcherExclude": {
    "**/.git/objects/**": true,
    "**/node_modules/**": true,
    "**/.pytest_cache/**": true,
    "**/__pycache__/**": true,
    "**/coverage_html/**": true
  }
}
```

**Benefits:**
- Code auto-formats on save
- Linting errors shown inline
- Run tests from VS Code UI
- Consistent team experience

**4. GitHub CLI Shortcuts**

```bash
# .github/gh-aliases.sh
# Developers can source this file for shortcuts

alias ghci="gh run list --limit 5"
alias ghview="gh run view"
alias ghpr="gh pr create --fill"
alias ghstatus="gh pr checks"

# Usage:
ghci  # View recent CI runs
ghpr  # Create PR with auto-filled template
ghstatus  # Check status of current PR
```

---

### 4.2 Documentation Requirements

**Recommendation: Comprehensive DevOps documentation in main repo + subset on Mintlify**

**Documentation Structure:**

**In Main Repository (`/docs/devops/`):**

```
docs/
├── devops/
│   ├── README.md                    # DevOps overview (start here)
│   ├── getting-started.md           # Quickstart for new contributors
│   ├── ci-cd-overview.md            # How CI/CD works
│   ├── local-development.md         # Running tests locally
│   ├── deployment-guide.md          # How to deploy to staging/production
│   ├── troubleshooting.md           # Common issues and solutions
│   ├── runbooks/
│   │   ├── rollback-procedure.md   # How to rollback deployments
│   │   ├── hotfix-process.md       # Emergency hotfix workflow
│   │   ├── incident-response.md    # What to do when things break
│   │   └── monitoring.md           # How to read dashboards/alerts
│   ├── architecture/
│   │   ├── ci-pipeline.md          # CI/CD architecture diagrams
│   │   ├── environments.md         # Staging vs production setup
│   │   └── infrastructure.md       # GPU servers, Docker, etc.
│   └── contributing/
│       ├── adding-tests.md         # How to write tests
│       ├── modifying-workflows.md  # How to change GitHub Actions
│       └── security.md             # Security best practices
```

**On Mintlify (Public Documentation):**

```
https://docs.arkos.dev/
├── Getting Started
│   ├── Installation
│   ├── Quickstart
│   └── Configuration
├── Development
│   ├── **Contributing Guide** ← Link to CI/CD info
│   ├── **Running Tests**
│   └── **Deployment Process** (high-level only)
├── API Reference
└── Troubleshooting
```

**What Goes on Mintlify:**
- ✅ High-level deployment process (for external contributors)
- ✅ How to run tests locally
- ✅ How to contribute (links to PR guidelines)
- ✅ Public-facing architecture overview
- ❌ Internal infrastructure details (security risk)
- ❌ Production credentials or server details
- ❌ Detailed runbooks (keep in private repo)

**Key Documentation Pages:**

**1. `docs/devops/getting-started.md`**

```markdown
# DevOps Getting Started Guide

Welcome to ARKOS development! This guide will help you set up your environment and understand our CI/CD pipeline.

## Prerequisites

- Python 3.11+
- Docker Desktop
- Git
- GitHub account with repo access

## Local Development Setup

### 1. Clone and Install Dependencies

\```bash
git clone https://github.com/SGIARK/arkos.git
cd arkos

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
\```

### 2. Run Tests Locally

\```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# View coverage report
open coverage_html/index.html
\```

### 3. Start Local SGLANG Instance

\```bash
# Start SGLANG Docker container
cd model_module
./run.sh

# Verify running
curl http://localhost:30000/v1/models
\```

### 4. Run Agent Locally

\```bash
# Start agent interface
python base_module/main_interface.py
\```

## CI/CD Workflow

### Making Changes

1. **Create feature branch**
   \```bash
   git checkout -b feature/my-new-feature
   \```

2. **Make changes and run checks locally**
   \```bash
   ./scripts/run-ci-locally.sh
   \```

3. **Commit changes** (pre-commit hooks run automatically)
   \```bash
   git add .
   git commit -m "feat: add my new feature"
   \```

4. **Push and create PR**
   \```bash
   git push origin feature/my-new-feature
   gh pr create --fill
   \```

5. **Wait for CI checks** (automatic)
   - Linting, formatting, type checks
   - Unit tests
   - Integration tests
   - Security scans

6. **Get reviews** (2 approvals required)

7. **Merge to main** (staging auto-deploys)

### Deploying to Staging

Staging deploys automatically when PR is merged to `main`.

To deploy your PR before merging:
\```bash
# Comment on PR:
/deploy-staging
\```

### Deploying to Production

Only maintainers can deploy to production:
1. Maintainer creates release tag from `main`
2. Manual approval required (2 maintainers)
3. Automated deployment to production
4. Health checks verify success

## Common Tasks

### Run Specific Tests
\```bash
pytest tests/unit/test_agent.py -v
\```

### Check Code Style
\```bash
black --check .
flake8 .
isort --check .
\```

### Auto-fix Formatting
\```bash
black .
isort .
\```

### View Staging Logs
\```bash
./scripts/view-logs.sh staging
\```

## Getting Help

- **CI/CD Issues:** Create issue with `ci-cd` label
- **Test Failures:** Check [Troubleshooting Guide](troubleshooting.md)
- **Deployment Questions:** Ask in #arkos-deployments Slack

## Next Steps

- [CI/CD Overview](ci-cd-overview.md)
- [Writing Tests](contributing/adding-tests.md)
- [Troubleshooting Guide](troubleshooting.md)
```

**2. `docs/devops/deployment-guide.md`**

```markdown
# ARKOS Deployment Guide

## Deployment Environments

| Environment | URL | Auto-Deploy | Purpose |
|------------|-----|-------------|---------|
| **Development** | localhost | No (manual) | Local development |
| **Staging** | staging.arkos.internal | Yes (on merge to `main`) | Integration testing |
| **Production** | arkos.mit.edu | No (manual approval) | Live user-facing service |

## Deployment Workflows

### Staging Deployment

**Trigger:** Automatic on merge to `main`

**Process:**
1. PR merged to `main`
2. GitHub Actions workflow triggered
3. Tests run (defensive check)
4. Docker image built
5. Deployed to staging server
6. Health checks verified
7. Slack notification sent

**Monitoring:**
\```bash
# Watch deployment
gh run watch

# Check staging health
curl http://staging.arkos.internal/health

# View logs
./scripts/view-logs.sh staging
\```

### Production Deployment

**Trigger:** Manual (maintainers only)

**Process:**
1. **Prepare Release (T-24 hours)**
   - Verify staging stable for 48+ hours
   - Complete QA checklist
   - Draft release notes

2. **Create Release Tag**
   \```bash
   git checkout main
   git pull
   git tag -a v0.2.0 -m "Release v0.2.0: [description]"
   git push origin v0.2.0
   \```

3. **Approve Deployment** (2 maintainers required)
   - GitHub Actions workflow triggered
   - Manual approval gate
   - 2 maintainers must approve

4. **Automated Deployment**
   - Backup production data
   - Deploy new version
   - Run health checks
   - Verify smoke tests

5. **Post-Deployment Monitoring** (15 minutes)
   - Watch error rates
   - Verify response times
   - Check user reports

6. **All-Clear or Rollback**
   - If successful: Announce in Slack
   - If issues: Initiate rollback

## Rollback Procedures

### Automatic Rollback

If health checks fail, automatic rollback is triggered:
\```yaml
# .github/workflows/deploy-production.yml
- name: Health check
  run: ./scripts/health-check.sh

- name: Auto-rollback on failure
  if: failure()
  run: ./scripts/rollback.sh ${{ github.event.before }}
\```

### Manual Rollback

\```bash
# Trigger manual rollback workflow
gh workflow run rollback-production.yml \
  -f target_version=v0.1.9 \
  -f reason="Critical bug in v0.2.0"

# Workflow will:
# 1. Validate target version exists
# 2. Deploy previous version
# 3. Verify health checks
# 4. Notify team
\```

See [Rollback Runbook](runbooks/rollback-procedure.md) for detailed steps.

## Hotfix Deployment

For critical production issues requiring immediate fix:

\```bash
# 1. Create hotfix branch from production tag
git checkout v0.2.0
git checkout -b hotfix/critical-bug-fix

# 2. Make minimal fix
# ... edit files ...
git commit -m "hotfix: fix critical bug"

# 3. Push and create PR
git push origin hotfix/critical-bug-fix
gh pr create --title "[HOTFIX] Critical bug fix" --label hotfix

# 4. Fast-tracked review (1 approval)
# 5. Merge and tag
git tag -a v0.2.1 -m "Hotfix v0.2.1"
git push origin v0.2.1

# 6. Expedited deployment (1 approval instead of 2)
\```

See [Hotfix Process](runbooks/hotfix-process.md) for full details.

## Deployment Checklist

### Pre-Deployment

- [ ] All CI checks passed
- [ ] Staging deployment successful (48+ hours ago)
- [ ] QA sign-off received
- [ ] Release notes drafted
- [ ] Stakeholders notified (24 hours notice)
- [ ] Rollback plan confirmed
- [ ] On-call team assigned

### During Deployment

- [ ] Code freeze announced (30 min before)
- [ ] Production backup completed
- [ ] Deployment initiated
- [ ] Health checks passed
- [ ] Smoke tests completed

### Post-Deployment

- [ ] Monitoring active (15 min intensive)
- [ ] No error spikes detected
- [ ] Response times normal
- [ ] User reports checked
- [ ] Code freeze lifted
- [ ] Team notified of success

## Troubleshooting Deployments

### Deployment Fails at Build Stage

\```bash
# Check workflow logs
gh run view --log

# Rebuild locally to debug
docker build -t arkos:debug .
\```

### Deployment Fails at Health Check

\```bash
# SSH to production server
ssh production.arkos.internal

# Check service status
systemctl status arkos
docker ps

# View recent logs
tail -n 100 /var/log/arkos/production.log

# Check SGLANG connectivity
curl http://localhost:30000/v1/models
\```

### Deployment Succeeds but Service Degraded

\```bash
# Check metrics dashboard
# → CPU, memory, GPU usage

# Check error logs
journalctl -u arkos -f --since "10 minutes ago"

# Consider rollback if errors increasing
\```

See [Troubleshooting Guide](troubleshooting.md) for more scenarios.

## Monitoring Post-Deployment

### Key Metrics to Watch

- **Response time:** <3s average
- **Error rate:** <1% of requests
- **GPU utilization:** 70-90% (not pegged at 100%)
- **Memory usage:** <80% of available

### Dashboards

- **Application Metrics:** [Grafana Dashboard](http://metrics.mit.edu/arkos)
- **Infrastructure:** [Server Monitoring](http://infra.mit.edu)
- **CI/CD Pipeline:** [GitHub Actions](https://github.com/SGIARK/arkos/actions)

## Security Considerations

- Never commit secrets to Git
- Use GitHub Secrets for sensitive values
- Rotate production credentials quarterly
- Audit production access logs monthly
- Enable 2FA for all maintainer accounts

## Contact

- **DevOps Team:** devops@arkos.mit.edu
- **On-Call:** See #arkos-deployments Slack topic
- **Emergency:** DM @devops-lead + @maintainer-lead
```

**3. `docs/devops/troubleshooting.md`**

```markdown
# ARKOS CI/CD Troubleshooting Guide

## Common CI Failures

### Test Failures

**Symptom:** Unit tests fail in CI but pass locally

**Possible Causes:**
1. Environment differences (Python version, dependencies)
2. Timezone or locale differences
3. Tests relying on local files not in repo

**Solutions:**
\```bash
# 1. Check Python version matches CI
python --version  # Should be 3.11

# 2. Run tests in Docker (matches CI environment)
docker run -it --rm -v $(pwd):/app python:3.11-slim /bin/bash
cd /app
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/

# 3. Check for hardcoded paths
grep -r "/Users/" tests/  # macOS paths
grep -r "C:\\\\" tests/    # Windows paths
\```

### Linting Failures

**Symptom:** flake8 or black errors in CI

**Solutions:**
\```bash
# Auto-fix formatting
black .
isort .

# Check remaining issues
flake8 .

# Common flake8 errors:
# E501: Line too long → Break into multiple lines
# F401: Unused import → Remove import
# E203: Whitespace before ':' → Let black handle it
\```

### Coverage Below Threshold

**Symptom:** Coverage check fails (below 80%)

**Solutions:**
\```bash
# 1. Generate coverage report
pytest tests/ --cov=. --cov-report=html

# 2. Open report and find uncovered lines
open coverage_html/index.html

# 3. Add tests for uncovered code
# or exclude deprecated/unreachable code

# 4. Verify coverage improved
pytest tests/ --cov=. --cov-report=term
\```

## Deployment Failures

### Staging Deployment Failed

**Symptom:** Workflow fails during staging deployment

**Check:**
\```bash
# 1. View workflow logs
gh run view --log

# 2. Check if staging server is accessible
ping staging.arkos.internal
ssh staging.arkos.internal "systemctl status arkos"

# 3. Check Docker image built successfully
docker pull ghcr.io/sgiark/arkos:latest

# 4. Check SGLANG is running
ssh staging.arkos.internal "docker ps | grep sglang"
\```

**Common Issues:**
- **SSH connection refused:** Check SSH keys in GitHub Secrets
- **Docker build failed:** Check Dockerfile syntax
- **Health check timeout:** SGLANG may be down

### Production Deployment Stuck

**Symptom:** Deployment waiting for approval but no one can approve

**Cause:** Required approvers not available

**Solution:**
\```bash
# Option 1: Wait for required approvers
# Option 2: Add emergency approver (DevOps team)
# Option 3: Cancel and reschedule deployment

# Cancel current deployment
gh run cancel <run-id>
\```

## Permission Issues

### Cannot Merge PR

**Symptom:** Merge button disabled

**Checks:**
- [ ] All required CI checks passed?
- [ ] Required number of approvals (2)?
- [ ] Branch up to date with main?
- [ ] Conversations resolved?

**Solutions:**
\```bash
# Update branch with latest main
git fetch origin
git merge origin/main
git push

# Re-run failed checks
# → Push empty commit to retrigger
git commit --allow-empty -m "chore: retrigger CI"
git push
\```

### Cannot Deploy to Staging

**Symptom:** `/deploy-staging` comment doesn't trigger workflow

**Possible Causes:**
1. Not a collaborator on repo
2. Workflow file has syntax error
3. GitHub Actions disabled

**Solutions:**
1. Ask maintainer to add you as collaborator
2. Validate workflow YAML:
   \```bash
   yamllint .github/workflows/deploy-pr-to-staging.yml
   \```

## Performance Issues

### CI Taking Too Long

**Symptom:** CI runs take >10 minutes

**Optimizations:**
\```yaml
# Cache dependencies in .github/workflows/ci.yml
- name: Cache pip
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

# Run jobs in parallel
jobs:
  lint:
    runs-on: ubuntu-latest
    steps: [...]

  test:
    runs-on: ubuntu-latest
    steps: [...]  # Runs parallel to lint
\```

### Tests Running Slowly Locally

**Solutions:**
\```bash
# Run only fast unit tests
pytest tests/unit/ -m "not slow"

# Skip integration tests
pytest tests/ --ignore=tests/integration

# Run tests in parallel
pytest tests/ -n auto  # Requires pytest-xdist
\```

## Log Access Issues

### Cannot View Staging Logs

**Symptom:** SSH connection refused

**Solutions:**
\```bash
# 1. Check if you have SSH access
ssh -T staging.arkos.internal

# 2. If no access, request via GitHub issue
# 3. Use GitHub Actions logs instead
gh run view --log

# 4. Check Grafana dashboard (if available)
\```

### Cannot Find Specific Error in Logs

**Solutions:**
\```bash
# Search logs for error
ssh staging.arkos.internal "grep -i 'error' /var/log/arkos/staging.log | tail -50"

# Search by timestamp
ssh staging.arkos.internal "grep '2025-11-01 15:' /var/log/arkos/staging.log"

# Search by module
ssh staging.arkos.internal "grep 'agent_module' /var/log/arkos/staging.log"
\```

## Getting Help

### When Stuck for >30 Minutes

1. **Search existing issues:**
   \```bash
   gh issue list --label "ci-cd"
   \```

2. **Ask in Slack:**
   - #arkos-dev for general questions
   - #arkos-deployments for deployment issues

3. **Create GitHub issue:**
   \```bash
   gh issue create --title "[CI] My specific problem" --label "ci-cd"
   \```

4. **Tag relevant people:**
   - CI/CD issues: @devops-team
   - Test failures: @maintainers
   - Deployment blocking production: @devops-team + @maintainers

### Escalation

If blocking production or time-sensitive:

1. Direct message in Slack: @devops-lead
2. Email: devops@arkos.mit.edu
3. For emergencies: Check #arkos-deployments Slack topic for on-call contact
\```

---

**Documentation Maintenance:**

- **Owner:** DevOps team maintains `/docs/devops/`
- **Review Cadence:** Quarterly (every 3 months)
- **Update Triggers:**
  - Workflow changes
  - Infrastructure changes
  - New common issues discovered
  - Post-mortem action items

**Documentation Quality Gates:**

- [ ] All runbooks tested on staging
- [ ] Screenshots/diagrams up to date
- [ ] Links not broken (use lychee link checker)
- [ ] Examples tested and working
- [ ] Reviewed by at least 1 contributor (for clarity)

---

## Summary & Implementation Priorities

**Phase 1 (Weeks 1-2): Foundation**
- [ ] Create `.github/workflows/ci.yml` (basic CI)
- [ ] Fix `requirements.txt` (add missing deps)
- [ ] Write `docs/devops/getting-started.md`
- [ ] Set up pre-commit hooks

**Phase 2 (Weeks 3-4): Testing & Deployment**
- [ ] Reorganize test structure
- [ ] Create `docker-compose.yml`
- [ ] Implement staging auto-deployment
- [ ] Write `docs/devops/deployment-guide.md`

**Phase 3 (Weeks 5-6): Production Readiness**
- [ ] Implement production deployment workflow
- [ ] Set up manual approval gates
- [ ] Create rollback automation
- [ ] Write runbooks

**Phase 4 (Weeks 7-8): Polish & Training**
- [ ] Add monitoring/alerting
- [ ] Create developer CLI tool
- [ ] Publish subset to Mintlify docs
- [ ] Conduct team training session

---

**Documentation Location Decision:**

**In Main Repo (`/docs/devops/`):**
- ✅ Detailed runbooks
- ✅ Internal architecture
- ✅ Troubleshooting guides
- ✅ Infrastructure details
- ✅ Security procedures

**On Mintlify (Public Docs):**
- ✅ High-level development workflow
- ✅ How to contribute (PR process)
- ✅ How to run tests locally
- ✅ Basic deployment overview
- ✅ Link to main repo docs for details

**Reasoning:**
- Main repo docs are version-controlled with code
- Sensitive infrastructure details stay private
- Public docs on Mintlify attract external contributors
- Clear separation of concerns
