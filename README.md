# MiniNeo: A Simplified Learned Query Optimizer with Tree Convolution

## Overview

Database query optimization is one of the most critical performance bottlenecks in modern data systems. Traditional query optimizers rely on hard-coded heuristics and statistical estimates that often fail for complex queries with multiple joins. MiniNeo addresses this challenge by applying machine learning to automatically discover optimal query execution strategies.

This project implements a streamlined version of the Neo learned query optimizer ([Marcus et al., 2019](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf)), specifically focusing on join order optimization using tree convolution neural networks. Unlike traditional optimizers that make decisions based on predetermined rules, MiniNeo learns from actual query execution performance to continuously improve its optimization strategies.

**Key Innovation**: MiniNeo uses tree convolution networks to process query execution plans as tree structures, enabling the system to automatically discover patterns in plan performance that traditional cost-based optimizers miss.

## Performance Results

Evaluated on the Join Order Benchmark (JOB) with 113 complex queries over the Internet Movie Database:

- **Consistent Improvement**: Achieves 20% average speedup (geometric mean) over PostgreSQL's optimizer
- **Peak Performance**: Up to 6x speedup for individual complex queries
- **Reliability**: Maintains performance at or above PostgreSQL baseline across all training iterations
- **Learning Efficiency**: Shows measurable improvements within the first few training iterations

The performance improvements are particularly significant for queries involving multiple joins—precisely the scenarios where traditional optimizers struggle most.

## Technical Approach

### Architecture Overview

MiniNeo operates through a two-phase learning system:

1. **Initial Training Phase**: Collects execution plans and performance data from PostgreSQL to bootstrap the learning process
2. **Runtime Optimization Phase**: Uses the trained neural network to guide plan search and continuously improves through execution feedback

### Core Components

**Query and Plan Representation**
- Encodes SQL queries as vectors capturing join graphs (adjacency matrices) and predicate information (one-hot vectors)
- Represents execution plans as tree structures with vectorized nodes for join operations and table scans
- Preserves structural information critical for pattern recognition

**Tree Convolution Network**
- Processes query execution plans while maintaining their inherent tree structure
- Uses specialized convolution filters that operate on parent-child-child "triangles" in the plan tree
- Captures local patterns such as inefficient join operator combinations and beneficial data access patterns
- Architecture includes query encoder, three tree convolution layers, dynamic pooling, and prediction layers

**Plan Search Algorithm**
- Implements best-first search guided by the neural network's value predictions
- Progressively builds complete execution plans by joining subtrees and specifying scan methods
- Bounded by configurable time limits (default 250ms) to ensure optimization doesn't become a bottleneck
- Falls back to greedy completion if time limit is reached

**Training Pipeline**
- Iterative improvement process that collects execution feedback and retrains the value network
- Starts with PostgreSQL plans to avoid cold-start problems common in reinforcement learning
- Continuously accumulates experience from both successful and failed optimization attempts

## Implementation Details

### Experimental Setup
- **Benchmark**: Join Order Benchmark (JOB) - 113 queries over Internet Movie Database
- **Baseline**: PostgreSQL 14.0 as both comparison optimizer and execution engine
- **Hardware**: Apple M1 Pro with 32GB RAM
- **Framework**: PyTorch for neural network implementation

### Training Process
- 20 training iterations with 100 epochs per iteration
- Batch size of 16, learning rate of 0.001
- Each iteration generates plans for all queries, executes them, and retrains the network
- Progressive improvement through accumulated experience

### Key Findings

**Learning Patterns**
- Geometric mean speedup shows more stable behavior than arithmetic mean, indicating consistent but modest improvements
- Performance variability between training runs demonstrates the system's ability to explore different optimization strategies
- Late-stage performance discoveries occur even after extended training, suggesting continued learning potential

**Optimization Strategies**
- Frequently selects different join orders than PostgreSQL, especially for multi-table queries
- Learns appropriate scan method selection based on predicate selectivity and join context
- Develops query-specific adaptations tailored to workload patterns

## Installation and Usage

### Requirements
- Python 3.9+
- PostgreSQL 14+
- PyTorch 1.10+
- Additional packages: numpy, pandas, psycopg2, matplotlib, tqdm

### Setup
```bash
git clone https://github.com/username/MiniNeo.git
cd MiniNeo
python -m venv minneo_env
source minneo_env/bin/activate
pip install -r requirements.txt
```

### Database Configuration
Create `.env` file with PostgreSQL credentials:
```
MINNEO_DB_HOST=localhost
MINNEO_DB_PORT=5432
MINNEO_DB_NAME=minneo_db
MINNEO_DB_USER=your_username
MINNEO_DB_PASSWORD=your_password
```

### Training Execution
```bash
# Download and setup IMDB dataset
python scripts/download_job_queries_direct.py
python scripts/dl_imdb.py

# Run training pipeline
python scripts/run_training_pipeline.py

# Evaluate performance
python scripts/evaluate.py
```

## Project Structure
```
MiniNeo/
├── src/
│   ├── config.py              # Configuration and settings
│   ├── db_utils.py            # Database utilities
│   ├── query_representation.py # Query encoding
│   ├── plan_representation.py  # Plan tree representation
│   ├── tree_convolution.py    # Tree convolution network
│   ├── plan_search.py         # Best-first search algorithm
│   └── training_pipeline.py   # Main training logic
├── scripts/
│   ├── run_training_pipeline.py
│   ├── evaluate.py
│   └── download_job_queries.py
├── data/                      # Datasets and benchmarks
└── models/                    # Saved neural network models and results
```

## Research Contributions

This implementation demonstrates several key insights:

1. **Simplified Architecture Effectiveness**: Focusing specifically on join order optimization while using tree convolution delivers significant performance improvements with manageable complexity

2. **Learning from Execution Feedback**: The system successfully learns from actual query performance rather than estimated costs, leading to more accurate optimization decisions

3. **Pattern Recognition in Query Plans**: Tree convolution networks effectively capture structural patterns in execution plans that correlate with performance

4. **Practical Machine Learning for Systems**: Shows how deep learning can be applied to core database components with measurable real-world benefits

## Limitations and Future Work

### Current Limitations
- Scope limited to join order optimization (not join algorithm selection or index usage)
- Training overhead requires executing many query plans during learning
- Workload-specific learning may not generalize to dramatically different query patterns
- Simplified query parsing may not handle all SQL syntax

### Future Directions
- Expand optimization scope to include join algorithm selection and index recommendations
- Develop techniques for better generalization across different workloads and schemas
- Implement more sample-efficient reinforcement learning to reduce training overhead
- Integrate more deeply with database internals for richer feedback signals

## Academic Foundation

This work builds on "Neo: A Learned Query Optimizer" by Marcus et al. (VLDB 2019), which pioneered the application of deep reinforcement learning to end-to-end query optimization. MiniNeo distills the core insights of Neo's approach while focusing on the specific problem of join order optimization.

The research demonstrates that machine learning approaches can practical alternatives to traditional query optimization techniques, particularly for complex analytical workloads where traditional optimizers often struggle.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{parr2024minineo,
  title={MiniNeo: A Simplified Learned Query Optimizer with Tree Convolution},
  author={Parr, Ian},
  year={2024},
  institution={University of Utah}
}
```