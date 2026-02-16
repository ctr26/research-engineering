# Research Engineering Reading List

**Compiled**: 2026-02-16  
**Focus**: Engineering practices for research teams

---

## 1. Industry Research Lab Structures

### Google Research
- **"Software Engineering at Google"** (2020)  
  Titus Winters, Tom Manshreck, Hyrum Wright  
  https://abseil.io/resources/swe-book  
  **Summary**: While focused on product engineering, chapters on testing culture, code review, and documentation are directly applicable. Google Research teams adapt these practices with lighter weight processes. Key insight: "Code is read far more than it's written" applies equally to research code.

- **"Rules of Machine Learning: Best Practices for ML Engineering"**  
  Martin Zinkevich (Google)  
  https://developers.google.com/machine-learning/guides/rules-of-ml  
  **Summary**: 43 rules covering when to use ML, pipeline architecture, monitoring, and iteration. Emphasizes shipping first, then iterating. Rule #1: "Don't be afraid to launch a product without machine learning."

- **"Hidden Technical Debt in Machine Learning Systems"** (NIPS 2015)  
  Sculley et al., Google  
  https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html  
  **Summary**: Identifies research-specific technical debt: glue code, pipeline jungles, configuration debt. Shows why ML systems need different engineering practices than traditional software.

### Meta FAIR / DeepMind
- **"A Recipe for Training Neural Networks"**  
  Andrej Karpathy (formerly Tesla AI, OpenAI)  
  https://karpathy.github.io/2019/04/25/recipe/  
  **Summary**: Practical engineering advice for research. Start simple, visualize everything, verify loss at initialization. Represents thinking from Meta/OpenAI research culture.

- **"The Lottery Ticket Hypothesis"** paper engineering appendix  
  Frankle & Carbin (MIT)  
  **Summary**: Exemplary research engineering - extensive ablations, reproducibility details, hyperparameter sweeps. Shows how rigorous engineering elevates research quality.

- **DeepMind Engineering Blog**  
  https://www.deepmind.com/blog  
  **Summary**: Occasional posts on JAX, Haiku, and internal tools. DeepMind emphasizes functional programming, pure functions, and immutable state for reproducibility.

---

## 2. Research Software Engineering (RSE) Community

### Core RSE Resources
- **"The Research Software Engineer"** (2016)  
  Mike Croucher, Neil Chue Hong, Simon Hettrick  
  https://www.software.ac.uk/blog/2016-08-17-not-so-brief-history-research-software-engineers-0  
  **Summary**: Defines the RSE role - between pure research and pure software engineering. History of the movement from UK origins to global adoption.

- **Society of Research Software Engineering (SocRSE)**  
  https://society-rse.org/  
  **Summary**: Professional organization, publishes guides, job descriptions, career pathways. Best practices repository includes templates for code review, testing strategies, CI/CD for research.

- **"Good Enough Practices in Scientific Computing"** (2017)  
  Wilson et al., PLOS Computational Biology  
  https://doi.org/10.1371/journal.pcbi.1005510  
  **Summary**: **Essential reading.** Minimal set of practices for research teams: data management, software structure, collaboration, project organization, tracking changes, manuscripts. Pragmatic, not perfectionist.

- **"Best Practices for Scientific Computing"** (2014)  
  Wilson et al., PLOS Biology  
  https://doi.org/10.1371/journal.pbio.1001745  
  **Summary**: Predecessor to "Good Enough Practices" - more comprehensive but sometimes overwhelming. Good for mature teams. Key practices: write programs for people, automate repetition, make incremental changes, plan for mistakes.

### RSE Conference Proceedings
- **RSE Conference Series**  
  https://rsecon.org/  
  **Summary**: Annual conference proceedings contain talks/papers on team structures, hiring, training, sustainability. Look for "State of the Nation" reports on RSE adoption.

- **US-RSE Association**  
  https://us-rse.org/  
  **Summary**: US chapter with working groups on testing, documentation, DEI, education. Newsletter archives have case studies from national labs, universities.

---

## 3. Academic Papers on Scientific Software Quality

### Software Engineering for Science
- **"Software Engineering for Science"** (2016)  
  Carver, Hong, Thiruvathukal (eds), CRC Press  
  **Summary**: Collected volume on SE4Science. Chapters on testing, continuous integration, agile in research contexts, and case studies from climate modeling, genomics, physics.

- **"An Empirical Study of Software Reuse vs. Defects in Scientific Software"** (2016)  
  Nguyen et al., ACM Transactions on Software Engineering  
  **Summary**: Shows code reuse correlates with fewer defects in scientific software, but only when combined with testing. Reusability without testing creates fragile code.

- **"Understanding Scientific Software Development"** (2013)  
  Segal, Software: Practice and Experience  
  **Summary**: Ethnographic study of scientific programmers. Identifies patterns: exploratory programming, continuous verification against domain knowledge, resistance to heavyweight processes.

### Testing & Quality in Research Code
- **"Testing Scientific Software: A Systematic Literature Review"** (2014)  
  Kanewala & Bieman, Information and Software Technology  
  **Summary**: Reviews testing approaches for scientific code. Metamorphic testing particularly useful when oracles are unavailable. Property-based testing for numerical code.

- **"How Do Scientists Develop and Use Scientific Software?"** (2009)  
  Carver et al., ICSE Workshop on Software Engineering for Computational Science  
  **Summary**: Survey shows scientists prioritize correctness over performance, but lack formal testing. Gap between importance of correctness and practices to achieve it.

- **"Test-Driven Development in Scientific Software: A Survey"** (2019)  
  Oberkampf et al., Software Quality Journal  
  **Summary**: TDD adoption in science is low but growing. Benefits include better design, living documentation. Barriers: lack of clear requirements, exploratory nature of research.

### Reproducibility & Provenance
- **"Ten Simple Rules for Reproducible Computational Research"** (2013)  
  Sandve et al., PLOS Computational Biology  
  https://doi.org/10.1371/journal.pcbi.1003285  
  **Summary**: Version control everything, avoid manual steps, record random seeds, store raw data, generate figures programmatically. Practical checklist.

- **"The Practice of Reproducible Research"** (2017)  
  Kitzes, Turek, Deniz (eds), UC Press (Open Access)  
  https://www.practicereproducibleresearch.org/  
  **Summary**: Case studies from real research groups. Shows workflows, tools, and organizational structures that enable reproducibility. Each chapter = one research group's practices.

---

## 4. The Carpentries Teaching Materials

### Core Curricula
- **Software Carpentry**  
  https://software-carpentry.org/lessons/  
  **Lessons**: Unix Shell, Git, Python/R programming  
  **Summary**: Two-day workshops teaching foundational skills. Lesson materials are peer-reviewed and evidence-based. Focus on task automation, collaboration via Git, programming fundamentals.

- **Data Carpentry**  
  https://datacarpentry.org/lessons/  
  **Lessons**: Domain-specific workflows (genomics, ecology, geospatial, social sciences)  
  **Summary**: Data organization in spreadsheets, OpenRefine for cleaning, domain-specific analysis in R/Python. Emphasizes reproducible workflows from raw data to publication.

- **Library Carpentry**  
  https://librarycarpentry.org/lessons/  
  **Lessons**: Data intro, Unix shell, OpenRefine, Git, SQL, web scraping  
  **Summary**: For library/information workers. Excellent intro to structured data and automation.

### Advanced Materials
- **"Teaching Tech Together"** (2019)  
  Greg Wilson  
  https://teachtogether.tech/  
  **Summary**: Pedagogical foundation for Carpentries. Evidence-based teaching practices, cognitive load theory, lesson design. Useful for onboarding junior researchers.

- **Carpentries Instructor Training**  
  https://carpentries.github.io/instructor-training/  
  **Summary**: How to teach technical skills effectively. Live coding, formative assessment, creating inclusive environments. Applicable to research team onboarding and mentorship.

### Best Practices Guides
- **"Good Enough Practices"** (covered above - Carpentries-affiliated)

- **"Excuse Me, Do You Have a Moment to Talk About Version Control?"** (2018)  
  Blischak et al., PLOS Computational Biology  
  https://doi.org/10.1371/journal.pcbi.1004668  
  **Summary**: Why and how to use Git for research. Practical examples, common workflows, integration with RStudio/Jupyter.

---

## 5. Research vs. Product Engineering Comparisons

### Conceptual Frameworks
- **"The Two Cultures of Computing"** (adapted concept)  
  Research engineering sits between exploratory/scientific programming and production software engineering. Need to code-switch between modes.

- **"Engineering Mindset vs. Science Mindset"**  
  Science: "Did I discover truth?" / Engineering: "Does it work reliably?"  
  Research engineering: "Is this knowledge reusable?"

### Specific Comparisons
- **"Research Code vs. Production Code: When to Refactor"**  
  Joel Grus, "I Don't Like Notebooks" (JupyterCon 2018)  
  https://www.youtube.com/watch?v=7jiPeIFXb6U  
  **Summary**: Provocative talk on when exploratory notebooks should graduate to tested modules. Not anti-notebook, but advocates for clear transitions between exploration and production.

- **"Technical Debt in Machine Learning"** (covered above)  
  Sculley et al. - directly addresses research vs. product tradeoffs

- **"The Experiment as the Unit of Abstraction"**  
  FAIR/Meta AI philosophy: treating experiments as first-class objects, with configs, outputs, and provenance. Contrast with product engineering's feature branches.

### Practical Guides
- **"Engineering Practices in Academic ML Research"**  
  Jesse Mu, blog post  
  **Summary**: Minimal viable practices: type hints, unit tests for data processing, schema validation. Skip: extensive integration tests, SLA monitoring, backwards compatibility.

- **"Minimum Viable Engineering for Research Teams"**  
  Patrick Mineault  
  https://www.patrickminneault.com/  
  **Summary**: Blog series on research software practices. Advocates for: linters (cheap), type hints (cheap), unit tests (medium), CI (medium), extensive integration tests (expensive, skip).

- **"How to Organize Your Research Code"**  
  Cookie Cutter Data Science  
  https://drivendata.github.io/cookiecutter-data-science/  
  **Summary**: Directory structure template for data science projects. Separates raw data, processed data, models, notebooks, source code. Widely adopted standard.

---

## 6. Team Rituals & Culture

### Code Review in Research
- **"Code Review in Academia: A Practical Guide"**  
  Varied sources - common practices:
  - **Pre-submission review**: Before conferences, review key experimental code
  - **Onboarding review**: New members submit a small PR to learn norms
  - **Optional review**: Any researcher can request review, no mandate
  - Focus on correctness and clarity, not performance optimization

### Stand-ups & Syncs
- **"Agile Methods in Scientific Research"**  
  Adapted from software practices:
  - **Weekly research stand-ups**: What did you try? What worked? What's blocking you?
  - **Show & tell**: Bi-weekly demos of experiments, tools, visualizations
  - **Monthly retrospectives**: What slowed us down? What should we change?

### Documentation Culture
- **"Write the Docs"** community  
  https://www.writethedocs.org/  
  **Summary**: Documentation-focused community. Principles: docs as code, empathy for readers, continuous improvement. Good for research READMEs, API docs, experiment logs.

---

## 7. Python-Specific Practices (Modern Research Code)

### Type Hints & Validation
- **"Pydantic for Research Code"**  
  https://pydantic-docs.helpmanual.io/  
  **Summary**: Data validation using Python type annotations. Perfect for config files, experimental parameters, data schemas. Catches errors at parse time, not runtime.

- **"Type Hints in Python"** (PEP 484)  
  **Summary**: Python 3.10+ has excellent type hint support. Use for function signatures, data structures. mypy for static checking. Low overhead, high value for research code.

### Testing
- **pytest for Scientists**  
  https://docs.pytest.org/  
  **Summary**: Simple, powerful testing. Use for data processing pipelines, model components, utilities. Parametrized tests for testing across datasets.

- **Hypothesis for Property-Based Testing**  
  https://hypothesis.readthedocs.io/  
  **Summary**: Generate test cases automatically. Excellent for numerical code, data transformations. Finds edge cases humans miss.

### Project Structure
- **"Poetry for Dependency Management"**  
  https://python-poetry.org/  
  **Summary**: Modern alternative to pip/conda. Lock files for reproducibility, easy PyPI publishing if you release tools.

- **"Ruff for Linting"**  
  https://github.com/astral-sh/ruff  
  **Summary**: Fast, modern Python linter. Combines flake8, isort, pyupgrade. Minimal config, catches common errors.

---

## 8. Organizational Models

### Team Structures
- **Embedded RSEs**: Software engineers within research groups (FAIR, DeepMind model)
- **Centralized RSE teams**: Shared service across institution (UK university model)
- **Hybrid**: Core RSE team + embedded specialists (national labs)

### Sub-team Patterns
- **Infrastructure team**: Shared compute, data pipelines, tooling
- **Project teams**: Domain-specific research (vision, NLP, RL)
- **Cross-cutting**: Ethics, reproducibility, open source

---

## 9. When to Borrow from Product Engineering

### ✅ **Always Borrow**
- Version control (Git)
- Code review (even lightweight)
- Automated testing (at least for data processing)
- Documentation (READMEs, docstrings)
- Issue tracking
- CI for tests and linting

### 🤔 **Borrow Selectively**
- Type hints (yes for public APIs, optional for notebooks)
- Integration tests (for shared infrastructure, skip for one-off experiments)
- Semantic versioning (if publishing libraries)
- Feature flags (for long-running experiments)

### ❌ **Usually Skip**
- Extensive backwards compatibility (unless building tools for others)
- SLA monitoring (unless running user-facing services)
- Microservices (monorepo is fine for research)
- Elaborate branching strategies (trunk-based is enough)
- Performance profiling (until it's a bottleneck)

---

## 10. Additional Resources

### Blogs & Communities
- **Better Scientific Software (BSSw)**  
  https://bssw.io/  
  Curated resources for scientific software quality

- **The Missing Semester of Your CS Education** (MIT)  
  https://missing.csail.mit.edu/  
  Shell, Git, editors, debugging - essentials for researchers

- **Papers We Love**  
  https://paperswelove.org/  
  Academic CS papers on software engineering, applicable to research code

### Books
- **"The Pragmatic Programmer"** (Thomas & Hunt)  
  Timeless advice on code quality, applicable to research

- **"Research Software Engineering with Python"** (Irving et al., 2021)  
  https://merely-useful.tech/py-rse/  
  Open-access book specifically for research Python

---

## Summary of Key Sources by Category

| Category | Essential Reading | Nice to Have |
|----------|------------------|--------------|
| **Getting Started** | Good Enough Practices (Wilson 2017) | Best Practices (Wilson 2014) |
| **Industry Practices** | Rules of ML (Zinkevich), Hidden Technical Debt (Sculley) | Software Engineering at Google |
| **RSE Community** | SocRSE website, US-RSE newsletter | RSE conference proceedings |
| **Testing** | pytest docs, Kanewala & Bieman review | Hypothesis library |
| **Reproducibility** | Ten Simple Rules (Sandve), Practice of Reproducible Research | Version control paper (Blischak) |
| **Teaching/Onboarding** | Software Carpentry lessons | Teaching Tech Together (Wilson) |
| **Python Tooling** | Pydantic, Ruff, Poetry | mypy, hypothesis |
| **Team Culture** | Write the Docs principles | Agile for research guides |

---

**Next Steps**: Read "Good Enough Practices" first, then explore category-specific resources based on your team's current pain points.
