"""Compatibility entry point for the modular background worker."""
from al_agent.background.config import *  # noqa: F401,F403
from al_agent.background.resources import *  # noqa: F401,F403
from al_agent.background.research import *  # noqa: F401,F403
from al_agent.background.maintenance import *  # noqa: F401,F403
from al_agent.background.runner import main

if __name__ == "__main__":
    main()
