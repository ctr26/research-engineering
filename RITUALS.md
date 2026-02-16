# Research Team Rituals

**Compiled**: 2026-02-16  
**Philosophy**: Lightweight, value-generating rituals that respect research autonomy while enabling collaboration

---

## Core Principles

1. **Respect deep work** - Minimize synchronous meetings; protect focus time
2. **Show, don't tell** - Demos > status updates
3. **Async-first** - Written updates scale better than meetings
4. **Optional by default** - Mandate only what's essential
5. **Iterate on rituals** - Drop what doesn't work, amplify what does

---

## Daily Rituals

### 🌅 Personal Check-in (Individual, 5 min)
**When**: Start of work day  
**Who**: Everyone  
**Format**: Personal ritual, not shared

**Template**:
```markdown
## 2026-02-16

### Yesterday
- Ran baseline experiments for paper X
- Debugging data pipeline issue
- Read 2 papers on attention mechanisms

### Today
- Fix data pipeline bug
- Start ablation study
- Review PR from teammate

### Blockers
- Waiting for compute cluster access
- None
```

**Why**: Clarifies priorities, surfaces blockers early. Keep it private or share in async channel.

**Tool**: Plain text file, Notion, Slack thread, whatever fits your flow.

---

### 📝 Commit Messages as Micro-Logs
**When**: Every commit  
**Format**: Meaningful commit messages

**Good**:
```
feat: Add attention visualization to model
fix: Correct off-by-one error in data loader
exp: Test lr=1e-4 with cosine schedule (baseline: 0.82 acc)
```

**Why**: Git log becomes experiment journal. `git log --oneline` shows research trajectory.

**Convention**:
- `feat:` - New feature/capability
- `fix:` - Bug fix
- `exp:` - Experiment run
- `docs:` - Documentation
- `refactor:` - Code cleanup

---

## Weekly Rituals

### 📊 Research Stand-up (Team, 30 min)
**When**: Monday or Wednesday morning  
**Who**: Core research team  
**Format**: Quick round-robin

**Structure** (5 min per person):
```
1. What did you try last week?
   - Experiments, papers, code
   
2. What worked? What didn't?
   - Results, insights, failures
   
3. What are you doing this week?
   - Immediate next steps
   
4. Any blockers or asks?
   - "Need GPU time"
   - "Can someone review this code?"
   - "Looking for feedback on paper draft"
```

**Rules**:
- ✅ Share results, not just activities
- ✅ Celebrate negative results (they're information!)
- ✅ Ask for help explicitly
- ❌ No problem-solving in standup (take offline)
- ❌ No status theater ("busy work" isn't progress)

**Why**: Coordination without micromanagement. Surfaces collaboration opportunities.

---

### 🎨 Show & Tell (Team, 45 min)
**When**: Bi-weekly, alternating with stand-up  
**Who**: Whole team + interested observers  
**Format**: Informal demos

**What to Show**:
- **New results**: "Here's what happened when I tried X"
- **Tools**: "I built a script to visualize attention weights"
- **Papers**: "This paper has a cool idea we should try"
- **Failed experiments**: "This didn't work, here's why"
- **Work-in-progress**: "I'm stuck on this problem, ideas?"

**Structure**:
```
1. Presenter shares screen (5-10 min)
   - Show the thing, not slides
   
2. Q&A and discussion (10-15 min)
   - "What if you tried...?"
   - "This reminds me of paper Y..."
   
3. Next steps (optional, 2 min)
   - "I'll try that suggestion"
   - "Let's sync offline"
```

**Why**: 
- Builds shared context
- Cross-pollinates ideas
- Makes failures visible (reduces duplication)
- Low-stakes feedback

**Rotation**: Everyone presents at least once per month. Sign up voluntarily.

---

### 📚 Paper Reading Club (Optional, 1 hour)
**When**: Weekly, fixed time  
**Who**: Anyone interested  
**Format**: Rotating facilitator

**Options**:

#### Option A: Deep Dive (One Paper)
```
Week before:
- Facilitator shares paper link
- Everyone reads independently

Meeting:
- 10 min: Facilitator summarizes key ideas
- 30 min: Discussion
  - What's novel?
  - What's unclear?
  - How does it relate to our work?
- 10 min: Next steps
  - Should we implement this?
  - Follow-up papers to read?
- 10 min: Pick next paper
```

#### Option B: Breadth (Multiple Papers)
```
Meeting:
- Each person presents 1 paper (5 min)
  - "Here's the idea in plain English"
  - "Why it's interesting"
- Open discussion (30 min)
```

**Why**: Keeps team current, sparks ideas. Optional attendance respects focus time.

---

## Monthly Rituals

### 🔄 Retrospective (Team, 1 hour)
**When**: Last Friday of the month  
**Who**: Core team  
**Format**: Structured reflection

**Agenda**:
```
1. Review team goals (5 min)
   - What did we want to accomplish this month?
   
2. What went well? (15 min)
   - Successes, breakthroughs, good processes
   - Action: Do more of this
   
3. What slowed us down? (15 min)
   - Blockers, frustrations, inefficiencies
   - Action: Fix, delegate, or accept
   
4. What should we change? (20 min)
   - Rituals: Keep, modify, or drop?
   - Tools: What's not working?
   - Processes: What's friction?
   
5. Action items (5 min)
   - Who will do what by when?
```

**Framework** (Mad/Sad/Glad):
```
😡 Mad: What frustrated us?
  - "Compute cluster was down 3 days"
  
😢 Sad: What disappointed us?
  - "Didn't finish paper revision"
  
😊 Glad: What made us happy?
  - "New model beats baseline!"
```

**Why**: Continuous improvement. Prevents rituals from becoming bureaucracy.

---

### 🎯 Planning (Team, 1 hour)
**When**: First Monday of the month  
**Who**: Core team + PM/lead  
**Format**: Roadmap review

**Agenda**:
```
1. Review last month's goals (10 min)
   - What shipped? What slipped?
   
2. Upcoming milestones (15 min)
   - Conference deadlines
   - Project deliverables
   - External dependencies
   
3. This month's focus (25 min)
   - Top 3 priorities per person
   - Resource allocation (compute, people)
   - Risks and dependencies
   
4. Commit to goals (10 min)
   - What are we committing to?
   - What are we explicitly NOT doing?
```

**Output**: Shared doc with monthly goals and owners.

**Why**: Alignment without over-planning. Research is unpredictable; adapt monthly.

---

## Quarterly Rituals

### 📈 Research Review (Team + Stakeholders, 2 hours)
**When**: End of quarter  
**Who**: Research team + leadership/advisors/collaborators  
**Format**: Formal presentation + discussion

**Structure**:
```
1. Accomplishments (30 min)
   - Papers published/submitted
   - New capabilities/tools
   - Key results
   
2. Insights (30 min)
   - What we learned (positive and negative)
   - Surprising findings
   - Dead ends explored
   
3. Roadmap (30 min)
   - Next quarter priorities
   - Resource needs
   - Risks
   
4. Open discussion (30 min)
   - Stakeholder feedback
   - Strategic direction
   - Collaborations
```

**Why**: Accountability, strategic alignment, stakeholder engagement.

---

### 🧹 Codebase Health Check (Team, half-day)
**When**: Quarterly  
**Who**: Whole team  
**Format**: Pair refactoring session

**Activities**:
```
1. Audit (1 hour)
   - What code is abandoned?
   - What's actively used?
   - What needs cleanup?
   
2. Prioritize (30 min)
   - High-value cleanups:
     - Shared utilities with no tests
     - Outdated dependencies
     - Broken documentation
   
3. Clean up (2 hours)
   - Pair programming: junior + senior
   - Delete dead code
   - Add tests to critical paths
   - Update README files
   
4. Document decisions (30 min)
   - What did we change?
   - What did we decide to keep/delete?
```

**Why**: Prevents technical debt accumulation. Teaches best practices through pairing.

---

## Ad-hoc Rituals

### 🚀 Pre-Submission Code Review
**When**: Before submitting paper with code  
**Who**: Author + 1-2 reviewers  
**Duration**: 1-2 hours

**Checklist**:
```
[ ] README explains how to reproduce key results
[ ] Dependencies are specified (requirements.txt or environment.yml)
[ ] Main experiments have clear entry points (scripts/train.py)
[ ] Random seeds are set for reproducibility
[ ] No hardcoded paths (use relative paths or configs)
[ ] Critical data processing has tests
[ ] Code matches paper description
[ ] Artifacts are organized (model checkpoints, figures, logs)
```

**Why**: Ensure code is reproducible before publication. Saves embarrassment.

---

### 🎓 Onboarding Review
**When**: Within first 2 weeks for new team member  
**Who**: New member + assigned buddy  
**Format**: First PR review

**Process**:
```
1. New member picks a small task:
   - Add a new metric to evaluation
   - Fix a bug from issue tracker
   - Add tests to a module
   
2. Opens PR

3. Buddy reviews, focusing on:
   - Team coding style
   - Testing practices
   - Documentation norms
   
4. Pair programming session to address feedback
```

**Why**: Teaches team norms through practice. Low-stakes learning.

---

### 🔥 Crisis Response
**When**: Major issue (critical bug, deadline crunch, compute outage)  
**Format**: Temporary war room

**Structure**:
```
1. Declare crisis (lead decision)
   - Clear end condition: "We're in crisis until X is resolved"
   
2. Daily sync (15 min)
   - What's unblocked?
   - What's still blocked?
   - Who needs help?
   
3. Suspend other rituals
   - Skip stand-ups, paper club
   - Focus all energy on crisis
   
4. Post-mortem after resolution
   - What went wrong?
   - How do we prevent it?
   - Update runbooks
```

**Why**: Focus energy during critical periods. Avoid burning out with permanent crisis mode.

---

## Asynchronous Rituals

### 📧 Weekly Written Updates
**When**: End of week  
**Who**: Everyone  
**Format**: Shared doc or Slack thread

**Template**:
```markdown
## Week of 2026-02-10

### Highlights
- Achieved 0.85 accuracy on benchmark X (prev: 0.82)
- Submitted PR #42 with data pipeline refactor
- Read 3 papers on efficient transformers

### Challenges
- Training diverged with lr > 1e-3, still debugging
- Blocked on cluster access for large-scale run

### Next Week
- Finish ablation study for Section 4
- Start writing related work section
- Review code from new team member
```

**Why**: 
- Async-first communication
- Creates searchable record
- Reduces meeting overhead

---

### 💬 Slack/Discord Best Practices

**Channels**:
```
#general         - Announcements, team-wide
#research        - Paper discussions, ideas
#results         - Experiment updates (auto-post from W&B?)
#code            - PR reviews, tech discussions
#random          - Non-work chat
#help            - "I'm stuck on X"
```

**Norms**:
```
✅ Use threads for discussions (keeps channel clean)
✅ Emoji react to acknowledge (reduce "thanks" messages)
✅ Post results with context: "Tried X, got Y, because Z"
✅ Share failures: "This didn't work, don't try it"
❌ Don't expect instant replies (async-first)
❌ Avoid DMing unless sensitive (keep knowledge public)
```

---

### 📂 Shared Documentation

**Wiki Structure**:
```
wiki/
├── onboarding/
│   ├── setup.md              # Dev environment setup
│   ├── codebase-tour.md      # Where things live
│   └── first-tasks.md        # Good starter tasks
├── runbooks/
│   ├── training-models.md    # How to run experiments
│   ├── debugging-guide.md    # Common issues
│   └── cluster-access.md     # Compute resources
├── decisions/
│   ├── 2026-01-15-use-pytorch.md   # ADRs (Architecture Decision Records)
│   └── 2026-02-01-experiment-tracking.md
└── references/
    ├── useful-papers.md      # Curated reading list
    └── tools.md              # Internal tools docs
```

**Decision Record Template** (ADR):
```markdown
# Use PyTorch for All Models

**Date**: 2026-01-15  
**Status**: Accepted  
**Deciders**: Alice, Bob, Carol

## Context
We need to choose a deep learning framework. Team has experience with both PyTorch and TensorFlow.

## Decision
Use PyTorch for all new models.

## Rationale
- Team prefers imperative style for research
- Better debugging experience
- Stronger ecosystem for NLP (our domain)

## Consequences
- Need to convert existing TF models (2 weeks of work)
- New hires should know PyTorch
- Can still use TF for production deployment if needed
```

---

## Sub-Team Interactions

### Infrastructure ↔ Research Teams

**Weekly Office Hours**:
```
When: Friday 2-3 PM
Who: Infrastructure team available
Format: Open Q&A

Research teams can:
- Request new tools/features
- Report bugs
- Get help with compute issues
```

**Monthly Roadmap Sync**:
```
Infrastructure shares:
- What's shipping this month
- Breaking changes (advance warning)
- Capacity constraints

Research teams share:
- Upcoming compute needs
- Pain points with current tools
- Feature requests
```

---

### Paper Writing Collaborations

**Pre-writing Kickoff** (1 hour):
```
1. Align on story (30 min)
   - What's the key contribution?
   - What's the narrative arc?
   - Who's the audience?

2. Divide sections (20 min)
   - Who writes what?
   - Who runs what experiments?
   
3. Set timeline (10 min)
   - Draft deadlines
   - Review rounds
   - Submission target
```

**Weekly Writing Sync** (30 min):
```
- Share progress
- Resolve inconsistencies
- Identify gaps (need more experiments?)
- Redistribute work if needed
```

**Review Rounds**:
```
Round 1: Internal review (team)
  - Focus: Technical correctness
  
Round 2: External review (advisor/collaborator)
  - Focus: Clarity, positioning
  
Round 3: Polish
  - Focus: Writing quality, figures
```

---

## Rituals to Avoid

### ❌ Daily Stand-ups
**Why**: Too frequent for research. Weekly is enough.

**Exception**: During final paper deadline push (temporary).

---

### ❌ Mandatory All-Hands
**Why**: Research teams are often async/distributed. Mandate only essentials.

**Better**: Monthly all-hands, optional attendance. Record for those who miss.

---

### ❌ Individual Status Reports to Manager
**Why**: Feels like surveillance. Use shared updates instead.

**Better**: Weekly written updates in team channel (transparent, searchable).

---

### ❌ Extensive Sprint Planning
**Why**: Research isn't predictable enough for 2-week sprints.

**Better**: Monthly planning with flexible execution.

---

### ❌ Code Review for Every Commit
**Why**: Too heavyweight for exploratory research code.

**Better**: Review code that will be shared or published. Optional review for everything else.

---

## Ritual Health Check

Ask quarterly:

```
For each ritual:

1. Does it generate value?
   - Does it surface information, build context, or drive decisions?
   
2. Is it the right frequency?
   - Too often = busywork
   - Too rare = loses continuity
   
3. Is attendance right-sized?
   - Too many people = wasted time
   - Too few = missing perspectives
   
4. Could it be async instead?
   - Meetings are expensive; writing is cheap
   
5. Do people show up engaged?
   - If not, kill it or redesign
```

**Kill criteria**:
- No one prepares
- Outcomes are vague
- Could've been an email
- People multi-task during it

---

## Customization by Team Size

### Small Team (2-5 people)
**Keep**:
- Weekly stand-up (15 min)
- Bi-weekly show & tell
- Monthly retro
- Ad-hoc code review

**Skip**:
- Formal planning meetings
- Separate paper club (discuss papers in stand-up)
- Written updates (you already know what everyone's doing)

---

### Medium Team (6-15 people)
**Keep**:
- Weekly stand-up
- Bi-weekly show & tell
- Monthly retro + planning
- Paper club (optional)
- Weekly written updates
- Sub-team syncs

**Consider**:
- Infrastructure office hours
- Quarterly research review

---

### Large Team (16+ people)
**Keep**:
- Sub-team stand-ups (6-8 people max)
- Monthly all-hands (replace stand-up)
- Show & tell (rotate sub-teams presenting)
- Quarterly reviews
- Extensive async communication

**Add**:
- Tech lead sync (coordination layer)
- Working groups for cross-cutting concerns (infra, ethics, open source)

---

## Sample Weekly Schedule

### Small Research Team (5 people)

```
Monday:
  10:00 AM - Weekly stand-up (30 min)
  
Wednesday:
  (Async work day - no meetings)
  
Friday:
  2:00 PM - Show & tell (every other week, 45 min)
  2:00 PM - Paper club (alternating weeks, 1 hour)
  
Daily:
  Individual check-ins (personal ritual)
  
End of Week:
  Written update (async, 15 min)
```

**Focus time**: 90%+ of the week is unscheduled deep work.

---

## Starting New Rituals

### Launch Pattern

```
Week 1: Propose
  - Share ritual template in team channel
  - "Let's try this for 4 weeks"
  
Weeks 2-5: Execute
  - Run the ritual as designed
  - Gather feedback in retro
  
Week 6: Decide
  - Keep, modify, or kill
  - If keep: make it official
```

**Rule**: All rituals are experiments. Nothing is permanent.

---

## Templates & Tools

### Ritual Facilitation Rotation
```markdown
# Stand-up Facilitators (Q1 2026)

| Week        | Facilitator |
|-------------|-------------|
| Feb 3-7     | Alice       |
| Feb 10-14   | Bob         |
| Feb 17-21   | Carol       |
| Feb 24-28   | David       |

Responsibilities:
- [ ] Send reminder day before
- [ ] Start on time
- [ ] Keep to time boxes
- [ ] Take notes (shared doc)
- [ ] Surface action items
```

---

### Meeting Note Template
```markdown
# Research Stand-up - 2026-02-16

**Attendees**: Alice, Bob, Carol, David  
**Facilitator**: Alice  
**Duration**: 28 minutes

## Updates

### Alice
- **Last week**: Baseline experiments for attention paper
- **Results**: 0.83 accuracy (target: 0.85)
- **This week**: Try multi-head attention variant
- **Blockers**: None

### Bob
- **Last week**: Data pipeline refactor
- **Results**: PR #42 merged, 2x faster loading
- **This week**: Add caching layer
- **Asks**: Code review from Carol?

### Carol
- **Last week**: Literature review on efficient transformers
- **This week**: Start implementing sparse attention
- **Blockers**: Waiting for GPU allocation

### David
- **Last week**: Paper writing (intro + related work)
- **This week**: Finish methods section
- **Asks**: Feedback on intro from team

## Action Items
- [ ] Carol: Review Bob's caching PR by Wed
- [ ] Alice: Share attention experiment code by Fri
- [ ] David: Send paper draft for feedback by Thurs
```

---

## Summary: Ritual Starter Kit

**Week 1 (Minimal)**:
- Daily individual check-ins
- Weekly team stand-up
- Ad-hoc code review

**Month 1 (Add)**:
- Bi-weekly show & tell
- Monthly retro
- Written weekly updates (if >5 people)

**Quarter 1 (Add)**:
- Paper club (optional)
- Monthly planning
- Quarterly review

**Evolve**: Drop what doesn't work. Rituals serve the team, not vice versa.

---

**Final Principle**: The best ritual is the one your team actually does. Start small, iterate, and optimize for focus time.
