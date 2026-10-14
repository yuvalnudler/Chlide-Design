"""Pipeline stages. The order is data, and it is validated as a test."""

from chlide.stages.base import Outcome, OrderError, Stage, StageResult, summarise, validate_order
from chlide.stages.terminal import ORDER, PREEXISTING, terminal_stages

__all__ = ['Stage', 'StageResult', 'Outcome', 'OrderError', 'validate_order', 'summarise',
           'ORDER', 'PREEXISTING', 'terminal_stages']
