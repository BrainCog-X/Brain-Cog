"""
Implements abstract class for meta-reinforcement learning environments.
"""

from typing import Generic, TypeVar, Tuple
import abc


ObsType = TypeVar('ObsType')


class MetaEpisodicEnv(abc.ABC, Generic[ObsType]):
    @property
    @abc.abstractmethod
    def max_episode_len(self) -> int:
        """
        Return the maximum episode length.
        """
        pass

    @abc.abstractmethod
    def new_env(self) -> None:
        """
        Reset the environment's structure by resampling
        the state transition probabilities and/or reward function
        from a prior distribution.

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def reset(self) -> ObsType:
        """
        Resets the environment's state to some designated initial state.
        This is distinct from resetting the environment's structure
            via self.new_env().

        Returns:
            initial observation.
        """
        pass

    @abc.abstractmethod
    def step(
        self,
        action: int,
        auto_reset: bool = True
    ) -> Tuple[ObsType, float, bool, dict]:
        """
        Step the env.

        Args:
            action: integer action indicating which action to take
            auto_reset: whether or not to automatically reset the environment
                on done. if true, next observation will be given by self.reset()

        Returns:
            next observation, reward, and done flat
        """
        pass
