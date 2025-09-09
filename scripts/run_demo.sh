cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
QUERY_NUM=""
NUM_QUERIES=1
SHOW_PLANS=true

# Parse command line arguments
while getopts "q:n:ph" opt; do
  case $opt in
    q)
      QUERY_NUM=$OPTARG
      ;;
    n)
      NUM_QUERIES=$OPTARG
      ;;
    p)
      SHOW_PLANS=false
      ;;
    h)
      echo "Usage: $0 [-q query_number] [-n num_queries] [-p] [-h]"
      echo "Options:"
      echo "  -q query_number   Run a specific JOB query number (e.g., 1a, 19b)"
      echo "  -n num_queries    Number of random queries to run if no specific query is specified"
      echo "  -p               Hide query plans (shorter output)"
      echo "  -h               Show this help message"
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

echo -e "${YELLOW}MiniNeo Live Demo Launcher${NC}"
echo -e "${YELLOW}==========================${NC}"

# Step 1: Set up the database
echo -e "\n${GREEN}Step 1: Setting up the IMDB database...${NC}"
bash scripts/setup_demo_db.sh

# Check if database setup was successful
if [ $? -ne 0 ]; then
    echo -e "\n${RED}Failed to set up database. Aborting.${NC}"
    exit 1
fi

# Step 2: Run the demo
echo -e "\n${GREEN}Step 2: Running MiniNeo demonstration...${NC}"
echo -e "${YELLOW}Starting in 3 seconds...${NC}"
sleep 3

# Clear the screen for a clean demo
clear

# Run the demo with arguments
python scripts/demo.py --query-num "$QUERY_NUM" --num-queries "$NUM_QUERIES" --show-plans "$SHOW_PLANS"

echo -e "\n${GREEN}Demo completed!${NC}"