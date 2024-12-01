from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import re
import logging
from dateparser import parse as date_parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFrame(Enum):
    """Defines different time frames for temporal search"""
    CUSTOM = "custom"        # User-defined time range
    ALL_TIME = "all_time"    # No time restrictions
    STRICT = "strict"        # Exact time boundaries
    FLEXIBLE = "flexible"    # Allows for decay-based scoring

@dataclass
class TemporalConstraints:
    """Represents temporal constraints for a search query"""
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    time_frame: TimeFrame
    
    @property
    def has_constraints(self) -> bool:
        """Check if any temporal constraints are set"""
        return self.start_date is not None or self.end_date is not None

@dataclass
class SearchParameters:
    """Complete set of search parameters combining UI and query inputs"""
    query: str
    ui_temporal: Optional[TemporalConstraints] = None
    nl_temporal: Optional[TemporalConstraints] = None
    strict_temporal: bool = True
    
    def get_effective_constraints(self) -> TemporalConstraints:
        """Determine final temporal constraints, prioritizing UI inputs"""
        if not self.ui_temporal and not self.nl_temporal:
            return TemporalConstraints(None, None, TimeFrame.ALL_TIME)
            
        if self.ui_temporal:
            return self.ui_temporal
            
        return self.nl_temporal
    
class TemporalQueryParser:
    """Enhanced temporal expression parser with UI integration support"""
    
    def __init__(self):
        # Define temporal pattern matching rules
        self.relative_patterns = {
            'numbered_period': r'(?:last|past|previous)\s+(\d+)\s+(day|week|month|year)s?',
            'single_period': r'last (day|week|month|year)',
            'specific_day': r'yesterday|today|tomorrow',
            'since_pattern': r'since\s+(.+?)(?=\s+and|\s+or|$)',
            'between_pattern': r'between\s+(.+?)\s+and\s+(.+?)(?=\s+and|\s+or|$)',
            'relative_time': r'(\d+)\s+(day|week|month|year)s?\s+ago'
        }
        
        self.informal_patterns = {
            'recent': timedelta(days=7),
            'latest': timedelta(days=3),
            'new': timedelta(days=1),
            'current': timedelta(days=1)
        }

    def parse_query(self, query: str) -> Tuple[str, TemporalConstraints]:
        """
        Parse temporal expressions from a query and return cleaned query with constraints.
        """
        now = datetime.now()
        start_date = None
        end_date = now
        working_query = query.lower()
        time_frame = TimeFrame.FLEXIBLE  # Default to flexible matching for NL queries

        # Process various temporal patterns
        for pattern_name, pattern in self.relative_patterns.items():
            matches = list(re.finditer(pattern, working_query))
            for match in matches:
                if pattern_name == 'numbered_period':
                    number = int(match.group(1))
                    period = match.group(2)
                    start_date = self._calculate_period_start(number, period)
                elif pattern_name == 'specific_day':
                    start_date, end_date = self._handle_specific_day(match.group(0))
                
                working_query = working_query.replace(match.group(0), '').strip()

        # Handle informal expressions
        for term, delta in self.informal_patterns.items():
            if term in working_query:
                start_date = now - delta
                working_query = re.sub(r'\b' + term + r'\b', '', working_query)

        return working_query.strip(), TemporalConstraints(start_date, end_date, time_frame)

    def _calculate_period_start(self, number: int, period: str) -> datetime:
        """Calculate start date based on period specification"""
        now = datetime.now()
        if period == 'day':
            return now - timedelta(days=number)
        elif period == 'week':
            return now - timedelta(weeks=number)
        elif period == 'month':
            return now - timedelta(days=number * 30)
        else:  # year
            return now - timedelta(days=number * 365)

    def _handle_specific_day(self, day_ref: str) -> Tuple[datetime, datetime]:
        """Handle specific day references"""
        now = datetime.now()
        if day_ref == 'yesterday':
            start = now - timedelta(days=1)
        elif day_ref == 'today':
            start = now
        else:  # tomorrow
            start = now + timedelta(days=1)
            
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end