class AgentPlanner:
    """
    Decides what to check for an incident.

    Plan = ordered list of tool steps:
        get_logs -> get_features -> detect_anomaly
        -> one runbook search per suspicious signal (heartbeat/replication/network/disk)
        -> one general runbook search built from the evidence ("$auto")
        -> find_similar_incidents
    The final "produce a conclusion" step is done by IncidentAgent.
    """

    KNOWN_TOOLS = (
        "get_incident", "get_logs", "get_features",
        "detect_anomaly", "search_runbook", "find_similar_incidents",
    )
    SIGNAL_QUERIES = {
        "heartbeat": "HDFS DataNode heartbeat failure",
        "replication": "HDFS block replication failure under-replicated blocks",
        "network": "HDFS network connection timeout failure",
        "disk": "HDFS disk failure no space left on device",
    }

    def __init__(self, log_limit=30, max_signal_searches=2, similar_top_k=3):
        for name, value in (("log_limit", log_limit), ("similar_top_k", similar_top_k)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if isinstance(max_signal_searches, bool) or not isinstance(max_signal_searches, int) \
                or max_signal_searches < 0:
            raise ValueError("max_signal_searches must be a non-negative integer.")
        self.log_limit = log_limit
        self.max_signal_searches = max_signal_searches
        self.similar_top_k = similar_top_k

    def create_plan(self, incident):
        """incident = dict returned by AgentTools.get_incident()."""
        if not isinstance(incident, dict) or not incident.get("incident_id"):
            raise ValueError("incident must be a dict containing 'incident_id'.")
        iid = incident["incident_id"]

        steps = [
            ("get_logs", {"incident_id": iid, "limit": self.log_limit},
             "Read the log lines, most severe first"),
            ("get_features", {"incident_id": iid}, "Get the numeric features of the window"),
            ("detect_anomaly", {"incident_id": iid}, "Check whether the ML model finds it abnormal"),
        ]

        signals = incident.get("signals") or {}
        suspicious = sorted(
            ((name, count) for name, count in signals.items()
             if count and name in self.SIGNAL_QUERIES),
            key=lambda item: item[1], reverse=True,
        )[: self.max_signal_searches]
        for name, count in suspicious:
            steps.append((
                "search_runbook", {"query": self.SIGNAL_QUERIES[name]},
                f"Check {name}: {count} WARN/ERROR/FATAL line(s) mention it",
            ))

        steps.append(("search_runbook", {"query": "$auto"},
                      "Search runbooks using the strongest evidence found so far"))
        steps.append(("find_similar_incidents",
                      {"incident_id": iid, "top_k": self.similar_top_k},
                      "Find windows with a similar profile"))

        return [
            {"step": i, "tool": tool, "args": args, "purpose": purpose}
            for i, (tool, args, purpose) in enumerate(steps, start=1)
        ]

    def select_tools(self, plan):
        """Distinct tool names the plan needs, in order. Rejects unknown tools."""
        if not isinstance(plan, list) or not plan:
            raise ValueError("plan must be a non-empty list of steps.")
        tools = []
        for step in plan:
            tool = step.get("tool") if isinstance(step, dict) else None
            if tool not in self.KNOWN_TOOLS:
                raise ValueError(f"Plan contains unknown tool: {tool!r}")
            if tool not in tools:
                tools.append(tool)
        return tools