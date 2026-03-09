from typing import Dict, Any, List


class TaskExtractor:

    def extract_user_description(self, history: List[Dict]) -> str:
        raise NotImplementedError

    def extract_reasoning_steps(self, history: List[Dict]) -> List[Dict]:
        raise NotImplementedError