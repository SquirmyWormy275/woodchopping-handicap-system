"""
Configuration constants for the Woodchopping Handicap System.

This module centralizes all configuration values, magic numbers, and system parameters
to improve maintainability and make it easier to adjust system behavior.
"""

from dataclasses import dataclass
from typing import Final


# =============================================================================
# AAA Competition Rules
# =============================================================================

@dataclass(frozen=True)
class Rules:
    """Australian Axemen's Association Competition Rules"""

    # Handicap mark constraints
    MIN_MARK_SECONDS: int = 3
    """Minimum handicap mark (front marker)"""

    MAX_TIME_LIMIT_SECONDS: int = 180
    """Maximum time limit for completion"""

    # Performance variance (critical for fairness)
    PERFORMANCE_VARIANCE_SECONDS: int = 3
    """Absolute performance variation (±3 seconds) for all competitors"""


# =============================================================================
# Data Requirements & Validation
# =============================================================================

@dataclass(frozen=True)
class DataRequirements:
    """Data validation thresholds and requirements"""

    # Historical data requirements
    MIN_HISTORICAL_TIMES: int = 3
    """Minimum historical times required for new competitors"""

    MIN_ML_TRAINING_RECORDS_TOTAL: int = 30
    """Minimum total records required for ML model training"""

    MIN_ML_TRAINING_RECORDS_PER_EVENT: int = 15
    """Minimum records per event (SB/UH) for event-specific models"""

    # Validation ranges
    MIN_DIAMETER_MM: int = 225
    """Minimum valid block diameter in millimeters"""

    MAX_DIAMETER_MM: int = 500
    """Maximum valid block diameter in millimeters"""

    MIN_VALID_TIME_SECONDS: float = 10.0
    """Minimum valid time (exclusive) - elite choppers can achieve sub-15s times"""

    MAX_VALID_TIME_SECONDS: float = 300.0
    """Maximum valid time (inclusive)"""

    # Outlier detection
    OUTLIER_IQR_MULTIPLIER: float = 3.0
    """IQR multiplier for outlier detection (3x IQR = extreme outliers only)"""

    # Confidence thresholds
    HIGH_CONFIDENCE_MIN_EVENTS: int = 5
    """Minimum events for HIGH confidence predictions"""

    MEDIUM_CONFIDENCE_MIN_EVENTS: int = 3
    """Minimum events for MEDIUM confidence predictions"""


# =============================================================================
# Machine Learning Configuration
# =============================================================================

@dataclass(frozen=True)
class MLConfig:
    """XGBoost ML model hyperparameters and settings"""

    # Model hyperparameters
    N_ESTIMATORS: int = 100
    """Number of boosting rounds (trees)"""

    MAX_DEPTH: int = 4
    """Maximum tree depth (optimized for small datasets)"""

    LEARNING_RATE: float = 0.1
    """Boosting learning rate"""

    RANDOM_STATE: int = 42
    """Random seed for reproducibility"""

    OBJECTIVE: str = 'reg:squarederror'
    """Loss function (regression with squared error)"""

    TREE_METHOD: str = 'hist'
    """Tree construction algorithm (histogram-based)"""

    # Cross-validation
    CV_FOLDS: int = 5
    """Number of folds for cross-validation"""

    # Prediction validation
    MIN_PREDICTION_TIME: float = 5.0
    """Minimum reasonable prediction time (seconds)"""

    MAX_PREDICTION_TIME: float = 300.0
    """Maximum reasonable prediction time (seconds)"""

    # Feature names
    FEATURE_NAMES: tuple = (
        'competitor_avg_time_by_event',
        'event_encoded',
        'size_mm',
        'wood_janka_hardness',
        'wood_spec_gravity',
        'competitor_experience'
    )

    # Event encoding
    EVENT_ENCODING_SB: int = 0
    """Standing Block event encoding"""

    EVENT_ENCODING_UH: int = 1
    """Underhand event encoding"""

    # Default wood properties (fallback values when species lookup fails)
    # Based on Eastern White Pine (S01) - common baseline species
    DEFAULT_JANKA_HARDNESS: float = 1690.0
    """Default Janka hardness if species not found (Eastern White Pine)"""

    DEFAULT_SPECIFIC_GRAVITY: float = 0.34
    """Default specific gravity if species not found (Eastern White Pine)"""


# =============================================================================
# Monte Carlo Simulation Configuration
# =============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo simulation parameters"""

    NUM_SIMULATIONS: int = 2_000_000
    """Number of race simulations to run for maximum statistical precision"""

    # Fairness assessment thresholds
    FAIRNESS_THRESHOLD_EXCELLENT: float = 0.02
    """Win rate spread threshold for 'Excellent' rating (2%)"""

    FAIRNESS_THRESHOLD_VERY_GOOD: float = 0.05
    """Win rate spread threshold for 'Very Good' rating (5%)"""

    FAIRNESS_THRESHOLD_GOOD: float = 0.10
    """Win rate spread threshold for 'Good' rating (10%)"""

    FAIRNESS_THRESHOLD_FAIR: float = 0.15
    """Win rate spread threshold for 'Fair' rating (15%)"""

    # Visualization settings
    VISUALIZATION_BAR_MAX_LENGTH: int = 40
    """Maximum length of text-based bar charts"""


# =============================================================================
# LLM Configuration (Ollama)
# =============================================================================

@dataclass(frozen=True)
class LLMConfig:
    """Ollama LLM settings for AI-enhanced predictions"""

    DEFAULT_MODEL: str = "qwen2.5:32b"
    """Default Ollama model (32B parameters for maximum mathematical precision)"""

    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    """Ollama API endpoint"""

    TIMEOUT_SECONDS: int = 120
    """Request timeout in seconds (increased for comprehensive analyses)"""

    MAX_RETRIES: int = 2
    """Maximum retry attempts for failed requests"""

    # Token limits for different use cases
    TOKENS_TIME_PREDICTION: int = 50
    """Token limit for single-number time predictions (fast responses)"""

    TOKENS_ANALYSIS_SHORT: int = 200
    """Token limit for short comparative analysis (3-4 sentences)"""

    TOKENS_FAIRNESS_ASSESSMENT: int = 5000
    """Token limit for comprehensive fairness assessment (detailed multi-paragraph analysis)"""

    TOKENS_PREDICTION_ANALYSIS: int = 1000
    """Token limit for comprehensive prediction method analysis (15-20 sentence multi-section analysis)"""


# =============================================================================
# File Paths & Excel Configuration
# =============================================================================

@dataclass(frozen=True)
class Paths:
    """File paths and Excel sheet names"""

    # Excel file
    EXCEL_FILE: str = "woodchopping.xlsx"
    """Main Excel workbook filename"""

    # Sheet names
    COMPETITOR_SHEET: str = "Competitor"
    """Competitor roster sheet name"""

    RESULTS_SHEET: str = "Results"
    """Historical results sheet name"""

    WOOD_SHEET: str = "wood"
    """Wood species properties sheet name"""


# =============================================================================
# Event Codes
# =============================================================================

@dataclass(frozen=True)
class EventCodes:
    """Valid event type codes"""

    STANDING_BLOCK: str = "SB"
    """Standing Block event code"""

    UNDERHAND: str = "UH"
    """Underhand event code"""

    VALID_EVENTS: tuple = ("SB", "UH")
    """List of all valid event codes"""


# =============================================================================
# Display & UI Configuration
# =============================================================================

@dataclass(frozen=True)
class DisplayConfig:
    """Display and user interface settings"""

    # Column widths for tables
    COMPETITOR_NAME_WIDTH: int = 35
    """Width for competitor name column"""

    TIME_COLUMN_WIDTH: int = 10
    """Width for time display columns"""

    # Separators
    SEPARATOR_LENGTH: int = 70
    """Length of separator lines"""

    # Formatting
    TIME_DECIMAL_PLACES: int = 1
    """Decimal places for time display"""


# =============================================================================
# Confidence Level Strings
# =============================================================================

class ConfidenceLevels:
    """Confidence level string constants"""

    HIGH: Final[str] = "HIGH"
    MEDIUM: Final[str] = "MEDIUM"
    LOW: Final[str] = "LOW"


# =============================================================================
# Instantiate Config Objects
# =============================================================================

# Create singleton instances for easy import
rules = Rules()
data_req = DataRequirements()
ml_config = MLConfig()
sim_config = SimulationConfig()
llm_config = LLMConfig()
paths = Paths()
events = EventCodes()
display = DisplayConfig()
confidence = ConfidenceLevels()


# =============================================================================
# Helper Functions
# =============================================================================

def get_event_encoding(event_code: str) -> int:
    """
    Get numeric encoding for event type.

    Args:
        event_code: Event code (SB or UH)

    Returns:
        0 for SB, 1 for UH

    Raises:
        ValueError: If event_code is not valid
    """
    event_upper = event_code.upper()
    if event_upper == events.STANDING_BLOCK:
        return ml_config.EVENT_ENCODING_SB
    elif event_upper == events.UNDERHAND:
        return ml_config.EVENT_ENCODING_UH
    else:
        raise ValueError(f"Invalid event code: {event_code}. Must be SB or UH")


def is_valid_event(event_code: str) -> bool:
    """
    Check if event code is valid.

    Args:
        event_code: Event code to validate

    Returns:
        True if valid, False otherwise
    """
    return event_code.upper() in events.VALID_EVENTS


def get_confidence_level(num_events: int) -> str:
    """
    Determine confidence level based on number of historical events.

    Args:
        num_events: Number of historical events for competitor

    Returns:
        Confidence level string (HIGH/MEDIUM/LOW)
    """
    if num_events >= data_req.HIGH_CONFIDENCE_MIN_EVENTS:
        return confidence.HIGH
    elif num_events >= data_req.MEDIUM_CONFIDENCE_MIN_EVENTS:
        return confidence.MEDIUM
    else:
        return confidence.LOW
