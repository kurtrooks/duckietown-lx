from typing import Tuple

import numpy as np


def PIDController(
        v_0: float,
        theta_ref: float,
        theta_hat: float,
        prev_e: float,
        prev_int: float,
        delta_t: float
) -> Tuple[float, float, float, float]:
    """
    PID performing heading control.
    Args:
        v_0:        linear Duckiebot speed (given).
        theta_ref:  reference heading pose.
        theta_hat:  the current estiamted theta.
        prev_e:     tracking error at previous iteration.
        prev_int:   previous integral error term.
        delta_t:    time interval since last call.
    Returns:
        v_0:     linear velocity of the Duckiebot
        omega:   angular velocity of the Duckiebot
        e:       current tracking error (automatically becomes prev_e_y at next iteration).
        e_int:   current integral error (automatically becomes prev_int_y at next iteration).
    """

    kp = 5.0
    ki = 0.2
    kd = 0.1

    e = theta_ref - theta_hat
    
    e_int = prev_int + e*delta_t
    if e_int > 2.0:
        e_int = 2.0
    elif e_int < -2.0:
        e_int = -2.0

    e_der = (e - prev_e)/delta_t

    omega = kp*e + ki*e_int + kd*e_der

    # Hint: print for debugging
    #print(f"\n\nDelta time : {delta_t} \nE : {np.rad2deg(e)} \nE int : {e_int} \nPrev e : {prev_e} \nTheta hat: {np.rad2deg(theta_hat)} \n")
    print("theta,theta_ref",theta_ref,theta_hat)
    print("PID",kp,ki,kd)
    print("Err",e,e_int,e_der)
    # ---
    return v_0, omega, e, e_int
