#!/usr/bin/env python3
"""
Aerodynamic parameter sets for the paramotor model.

Reference
---------
J. Umenberger and A. H. Goktogan, "Guidance, Navigation and Control of a
Small-Scale Paramotor", ACRA 2012 (pap151.pdf).  Table 1 and section 3.2.

Two sets, IDENTICAL KEYS, so the same aero layer runs either one:

  PAPER_ACRA2012   the paper's own 1.55 kg / 2.15 m aircraft.  Used ONLY to
                   replicate the published figures and prove the aero layer is
                   correct.  Never flown on this robot.

  PEEK_1M          this robot.  0.325 kg / 1.00 m.

THE TWO AIRCRAFT ARE NOT SCALE MODELS OF EACH OTHER.  They differ independently
and in different directions in aspect ratio, wing loading, relative density and
mass distribution:

    relative density  mu = m/(rho A b)    0.507 -> 1.355   (2.67x)
    Froude            V^2/(g b)           1.691 -> 4.516   (2.67x)
    Reynolds          at mean chord       2.15e5 -> 0.87e5 (0.41x)
    aspect ratio      b^2/A               3.985 -> 5.100   (1.28x)

so IDENTICAL coefficients do NOT give identical dynamics.  With the same
C_lp = -0.127 the non-dimensional roll subsidence eigenvalue

    lambda_hat = C_lp / (8 mu i_xx)

is -0.667 for the paper and -0.190 here: 3.5x weaker damping in non-dimensional
time.  See paramotor_model_alignment.typ section 2.

Every entry below is tagged with how it was obtained:

    [PAPER]    taken from Table 1 / section 3.2 unchanged
    [AR]       paper value, corrected for the aspect-ratio change
    [ROBOT]    this vehicle's own geometry or hardware; the paper's value is
               not applicable at any scale factor
    [PROVISIONAL]  a placeholder that MUST be identified on the real vehicle.
               Flying on these is fine; sizing anything on them is not.
"""
import math

# ----------------------------------------------------------------------------
# The paper's aircraft.  Replication target only.
# ----------------------------------------------------------------------------
PAPER_ACRA2012 = dict(
    name="Umenberger & Goktogan 2012, Table 1",
    rho=1.225,                      # [PAPER] kg/m^3

    # --- parafoil geometry ---
    AP=1.16,                        # [PAPER] m^2, planform
    b=2.15,                         # [PAPER] m, span
    c=0.54,                         # [PAPER] m, chord
    chi=math.radians(20.0),         # [PAPER] rigging pitch, canopy w.r.t. body

    # --- parafoil lift and drag, eq. (15) ---
    CL0=0.4,                        # [PAPER]
    CLa=2.0,                        # [PAPER] per rad
    CD0=0.15,                       # [PAPER]
    CDa=1.0,                        # [PAPER] per rad^2

    # --- fuselage drag, eq. (10) ---
    AF=0.5,                         # [PAPER] m^2, reference area
    CD0F=0.15,                      # [PAPER]
    CDaF=1.0,                       # [PAPER]

    # --- pure aerodynamic moments, eq. (18) ---
    Clp=-0.127,                     # [PAPER] identified, section 3.2
    Clphi=-0.0055,                  # [PAPER] identified, section 3.2
    Cmq=-2.0,                       # [PAPER] Table 1
    Cm0=0.018,                      # [PAPER] Table 1
    Cma=-0.2,                       # [PAPER] Table 1
    Cnr=-0.0035,                    # [PAPER] identified; NOT in Table 1

    # --- validity envelope.  NOT in the paper: eq. (15) is linear with no
    #     stall, so C_L just keeps climbing.  Ours, deliberately. ---
    alpha_min=math.radians(-8.0),
    alpha_max=math.radians(18.0),
)

# ----------------------------------------------------------------------------
# This robot.
# ----------------------------------------------------------------------------
_B = 1.000                       # m, flat span         (build_paramotor.SPAN_FLAT)
_AR = 5.1                        # flat aspect ratio    (build_paramotor.AR_FLAT)
_S = _B ** 2 / _AR               # 0.19608 m^2
_C = _S / _B                     # 0.19608 m, constant-chord approximation
_CLA = 2.08
_E = 0.9                        # assumed span efficiency, not identified
_PAPER_AR = PAPER_ACRA2012["b"] ** 2 / PAPER_ACRA2012["AP"]
_CDA = (PAPER_ACRA2012["CDa"]
        - PAPER_ACRA2012["CLa"] ** 2 / (math.pi * _E * _PAPER_AR)
        + _CLA ** 2 / (math.pi * _E * _AR))

PEEK_1M = dict(
    name="1 m PEEK paramotor",
    rho=1.225,                      # [PAPER] sea level, same air

    # --- parafoil geometry: this vehicle's own, never the paper's scaled ---
    AP=_S,                          # [ROBOT] 0.19608 m^2
    b=_B,                           # [ROBOT] 1.000 m
    c=_C,                           # [ROBOT] 0.19608 m
    # Rigging pitch is EMERGENT here, not a parameter: the canopy is its own
    # free body and its attitude is set by the suspension line lengths.  The
    # paper needs chi=20deg only because it welds canopy and pod into one rigid
    # body.  Applying it again would double-count the rigging.
    chi=0.0,                        # [ROBOT]

    # --- parafoil lift and drag ---
    # Lifting line: invert C_La = a0/(1 + a0/(pi e AR)) at the paper's AR=3.985
    # with e=0.9 to get a0=2.432, re-evaluate at AR=5.1.
    CL0=0.4,                        # [PAPER] weakly AR-dependent
    CLa=_CLA,                       # [AR] 2.0 -> 2.08
    # With the updated lift slope the induced alpha^2 term is 0.300, not
    # 0.277 (which used the old slope). Keep the assumed residual term fixed.
    # This decomposition is provisional; it is not a measured drag polar.
    CDa=_CDA,                       # [AR] 1.0 -> approximately 0.945
    # Re is 2.5x lower (0.87e5), where a 125 um single skin is dominated by
    # laminar separation, and this also absorbs line drag, which differs.
    CD0=0.15,                       # [PROVISIONAL] identify on the vehicle

    # --- fuselage drag ---
    # Only the PRODUCT CD0F*AF is physical. The paper's reference drag area
    # is 0.075 m^2 on its larger model aircraft. The pod is a 140 x 110 mm box, so
    # 0.0154 m^2 frontal at a bluff-body CD of about 1.0 -> 0.0154 m^2.
    AF=0.140 * 0.110,               # [ROBOT] 0.0154 m^2
    CD0F=1.0,                       # [PROVISIONAL] assumed bluff-box drag
    CDaF=1.0,                       # [PAPER]

    # --- pure aerodynamic moments ---
    # Clp and Clphi are used ONLY in mode="lumped".  In the default strip mode
    # the roll moment comes from the panel force distribution and these rows are
    # suppressed (see pure_moments_frd(roll_native=True)).  Strip theory over
    # the arch generates roll damping; its magnitude depends on the force
    # reference and section-polar approximation. See test_aero.py for probes.
    Clp=-0.127,                     # [PAPER] lumped mode only
    # CORRECTION to an earlier note here.  This was set to 0 on the argument
    # that the suspension tendons produce the roll restoring moment
    # structurally.  They do not: the tendons produce the GRAVITY pendulum,
    # canopy displaced relative to pod.  The arc's AERODYNAMIC dihedral effect
    # -- sideslip over a curved lifting surface -- is a separate mechanism.
    # Strip mode generates this coupling from geometry. Earlier negative
    # C_lbeta reports used FLU-leftward velocity; FRD beta has the opposite
    # sign. Assess full coupled stability with flight tests, not this term alone.
    Clphi=0.0,                      # [ROBOT] lumped mode only; see above
    Cmq=-2.0,                       # [PAPER] set by l_P/b, which agrees to 16%
    # Cm0 and Cma are set by where the lines attach relative to the aerodynamic
    # centre.  This robot's A/B/C rows are its own.  Start here, then trim.
    Cm0=0.018,                      # [PROVISIONAL]
    Cma=-0.2,                       # [PROVISIONAL]
    # Strip theory also produces yaw damping natively -- measured C_nr = -0.038
    # against the -0.0035 identified in the paper, an 11x over-prediction that
    # is much less trustworthy than its roll damping.  The explicit term below
    # is kept (it adds about 8% on top) because it came from real flight data,
    # but C_nr is an identification target and the overlap is acknowledged.
    Cnr=-0.0035,                    # [PROVISIONAL] overlaps strip; see above

    alpha_min=math.radians(-8.0),
    alpha_max=math.radians(18.0),
)

# Brake coefficients (C_Ldelta_a, C_Ddelta_a, C_ldelta_a, C_ndelta_a) and the
# brake length d are DELIBERATELY ABSENT.  Section 3.5 of the plan is not yet
# implemented: the paper's Table 1 gives C_ldelta_a = +0.0021 while its own
# section 3.2 identifies -0.2959, a factor of 140 and a sign flip, and those
# were identified for a d/b = 0.186 brake cascade spread along the trailing
# edge.  This robot pulls one tendon at one corner of one tip panel.  The
# numbers cannot transfer; brake actuation currently acts only through the
# physical servo arms and tendons.

SETS = {"paper": PAPER_ACRA2012, "peek": PEEK_1M}
