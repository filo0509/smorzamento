#!/bin/bash
# Run analysis tools for smorzamento project

# Define colors for console output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if the script is run from the correct directory
if [ ! -d "scripts" ] || [ ! -d "data_raw" ] || [ ! -d "data_simulated" ]; then
    echo -e "${RED}Error: This script must be run from the smorzamento project root directory.${NC}"
    exit 1
fi

# Make sure results directories exist
mkdir -p results/plots
mkdir -p results/analysis

print_help() {
    echo -e "${BLUE}Smorzamento Analysis Tools${NC}"
    echo -e "Usage: ./run_analysis.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  linear       Run analysis for linear damping (exponential decay)"
    echo "  quadratic    Run analysis for quadratic damping (1/t decay)"
    echo "  long         Run analysis for long-duration data files"
    echo "  auto         Run automated analysis with reduced interaction"
    echo "  generate     Generate new simulated data files"
    echo "  all          Run all analysis tools sequentially"
    echo "  help         Display this help message"
    echo ""
}

run_linear_analysis() {
    echo -e "${GREEN}Running Linear Damping Analysis (Exponential Decay)...${NC}"
    python scripts/analyze_exponential_damping.py
}

run_quadratic_analysis() {
    echo -e "${GREEN}Running Quadratic Damping Analysis (1/t Decay)...${NC}"
    python scripts/analyze_quadratic_damping.py
}

run_long_analysis() {
    echo -e "${GREEN}Running Long-Duration Data Analysis...${NC}"
    python scripts/analyze_long_quadratic_damping.py
}

run_auto_analysis() {
    echo -e "${GREEN}Running Automated Analysis...${NC}"
    python scripts/analyze_quadratic_damping_auto.py
}

generate_data() {
    echo -e "${GREEN}Generating Simulated Data Files...${NC}"
    echo -e "${YELLOW}Choose data generation option:${NC}"
    echo "  1. Standard quadratic damping data"
    echo "  2. Multiple parameter configurations"
    echo "  3. Long-duration (100s) data"
    echo "  4. Return to main menu"
    echo -n "Enter option [1-4]: "
    read option
    
    case $option in
        1) python scripts/generate_standard_quadratic_damping.py ;;
        2) python scripts/generate_multiple_quadratic_damping.py ;;
        3) python scripts/generate_long_quadratic_damping.py ;;
        4) return ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
}

run_all() {
    echo -e "${GREEN}Running All Analysis Tools Sequentially...${NC}"
    run_linear_analysis
    run_quadratic_analysis
    run_long_analysis
    run_auto_analysis
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments provided, show interactive menu
    while true; do
        echo -e "${BLUE}Smorzamento Analysis Tools${NC}"
        echo -e "${YELLOW}Select an analysis option:${NC}"
        echo "  1. Linear damping analysis (exponential decay)"
        echo "  2. Quadratic damping analysis (1/t decay)"
        echo "  3. Long-duration data analysis"
        echo "  4. Automated analysis"
        echo "  5. Generate simulated data"
        echo "  6. Run all analysis tools"
        echo "  7. Exit"
        echo -n "Enter option [1-7]: "
        read option
        
        case $option in
            1) run_linear_analysis ;;
            2) run_quadratic_analysis ;;
            3) run_long_analysis ;;
            4) run_auto_analysis ;;
            5) generate_data ;;
            6) run_all ;;
            7) echo -e "${GREEN}Exiting...${NC}"; exit 0 ;;
            *) echo -e "${RED}Invalid option${NC}" ;;
        esac
        echo ""
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read
        clear
    done
else
    # Execute specific tool based on argument
    case $1 in
        linear) run_linear_analysis ;;
        quadratic) run_quadratic_analysis ;;
        long) run_long_analysis ;;
        auto) run_auto_analysis ;;
        generate) generate_data ;;
        all) run_all ;;
        help|--help|-h) print_help ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; print_help; exit 1 ;;
    esac
fi