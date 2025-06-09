
from DHKimTests.legged_gym.envs.base.legged_robot_field import LeggedRobotField
from DHKimTests.legged_gym.envs.base.legged_robot_noisy import LeggedRobotNoisyMixin

class RobotFieldNoisy(LeggedRobotNoisyMixin, LeggedRobotField):
    """ Using inheritance to combine the two classes.
    Then, LeggedRobotNoisyMixin and LeggedRobot can also be used elsewhere.
    """
    pass
