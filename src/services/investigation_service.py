import re
from collections import OrderedDict, deque

from src.agents.agent_tools import to_jsonable


class InvestigationService:
    """Starts and manages investigations: API -> InvestigationService -> IncidentAgent."""

    def __init__(self, incident_agent, cache=True, max_cache=200, max_history=100):
        if incident_agent is None:
            raise ValueError("incident_agent is required.")
        self.agent = incident_agent
        self.cache_enabled = cache
        self.max_cache = max_cache
        self._cache = OrderedDict()
        self._history = deque(maxlen=max_history)

    @staticmethod
    def _cache_key(incident_id):
        text = str(incident_id).strip()
        match = re.match(r"^(?:INC[-_ ]?)?0*(\d+)$", text, re.IGNORECASE)
        return f"INC-{int(match.group(1)):03d}" if match else text.upper()

    def investigate(self, incident_id, refresh=False):
        """Run (or reuse) an investigation. Raises IncidentNotFoundError / ValueError for bad ids."""
        key = self._cache_key(incident_id)
        if self.cache_enabled and not refresh and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        report = to_jsonable(self.agent.investigate(incident_id))
        self._remember(report)

        if self.cache_enabled:
            self._cache[key] = report
            while len(self._cache) > self.max_cache:
                self._cache.popitem(last=False)
        return report

    def analyze(self, query):
        report = to_jsonable(self.agent.analyze(query))
        self._remember(report)
        return report

    def get_history(self, limit=20):
        """Most recent investigations first."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        return list(self._history)[::-1][:limit]

    def clear_cache(self):
        self._cache.clear()

    def _remember(self, report):
        self._history.append({
            "incident_id": report.get("incident_id"),
            "query": report.get("query"),
            "root_cause": report.get("root_cause"),
            "confidence": report.get("confidence"),
            "anomaly_score": report.get("anomaly_score"),
            "generated_at": report.get("generated_at"),
        })