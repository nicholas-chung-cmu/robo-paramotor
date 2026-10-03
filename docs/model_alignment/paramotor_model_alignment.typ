#set document(title: "Porting the Umenberger–Göktoğan 6DOF Paramotor Model to a 0.33 kg Robot",
              author: "Paramotor modelling notes")
#set page(paper: "a4", margin: (x: 2.0cm, y: 2.2cm), numbering: "1")
#set text(font: ("Libertinus Serif", "New Computer Modern", "Times New Roman"), size: 10pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading: it => block(above: 1.1em, below: 0.6em)[#it]
#set math.equation(numbering: "(1)")
#show raw.where(block: false): it => box(fill: luma(240), inset: (x: 2pt), outset: (y: 2pt), radius: 1.5pt)[#it]

#let warn(body) = block(fill: rgb("#fdf2f2"), stroke: (left: 2pt + rgb("#c0392b")),
                        inset: 8pt, radius: 2pt, width: 100%)[#body]
#let note(body) = block(fill: luma(245), inset: 8pt, radius: 3pt, width: 100%)[#body]

#align(center)[
  #text(15pt, weight: "bold")[Porting the Umenberger–Göktoğan 6DOF paramotor model\ to the 0.33 kg PEEK robot]
  #v(0.2em)
  #text(9.5pt)[A change plan for `paramotor_mjcf/build_paramotor.py` → `paramotor.xml`]
  #v(0.2em)
  #text(9pt, style: "italic")[Reference: J. Umenberger and A. H. Göktoğan, “Guidance, Navigation and Control of a Small-Scale Paramotor,” ACRA 2012 (`pap151.pdf`).]
]

#v(0.5em)

= The governing constraint: these are not scale models of each other

#warn[
*Nothing dimensional in Table 1 of the paper transfers to this robot, and no
scaling law repairs that.* The paper's aircraft is 1.55 kg on a 2.15 m wing.
This robot is 0.325 kg on a 1.0 m wing, with a 0.400 kg ceiling. Those two
vehicles are not geometrically similar, not dynamically similar, and not
Reynolds-similar — they differ in aspect ratio, in wing loading, in relative
density and in mass distribution, independently and in different directions.

The paper's value here is its *model structure* — the functional form of
eqs. (8)–(22) — and its *identification method* (§3.2). Its numbers are worth
only their order of magnitude, and several are worth less than that.
]

Every parameter in Table 1 falls into one of three classes, and the whole plan
is organised around the distinction:

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(170),
  inset: 6pt,
  table.header([*Class*], [*Parameters*], [*Transfers?*]),
  [*I. Dimensional*],
  [$m^B$, $[I_B^B]^B$, $A^P$, $A^F$, $b$, $c$, $d$, $[s_(P B)]$, $[s_(F B)]$, $[s_(M F)]$, $[s_(C F)]$],
  [*Never.* The robot already knows all of these — from its own BOM, measured or
   budgeted. Read them out of the compiled model; do not scale the paper's.],
  [*II. Non-dimensional,\ geometry-driven*],
  [$C_(L_0)^P$, $C_(L_alpha)^P$, $C_(D_alpha)^P$, $C_(l_p)$, $C_(m_q)$, $C_(n_r)$, $chi$],
  [*Provisionally*, with an aspect-ratio correction. These are the real
   deliverable of the paper.],
  [*III. Non-dimensional,\ configuration-driven*],
  [$C_(D_0)^P$, $C_(D_0)^F$, $C_(D_alpha)^F$, $C_(l_phi)$, $C_(m_0)$, $C_(m_alpha)$,
   $C_(L_(delta_a))$, $C_(D_(delta_a))$, $C_(l_(delta_a))$, $C_(n_(delta_a))$],
  [*No.* Each depends on something this robot does differently: Reynolds number,
   line rigging, or brake cascade geometry. Structure transfers, values must be
   re-identified.],
)

= Similarity audit

All robot figures below are *measured from the compiled `paramotor.xml`*, not
assumed. The composite inertia is accumulated over all 30 geoms about the system
CG; the CG sits at $z = 0.1256$ m, i.e. 0.110 m above the pod CG and 0.415 m
below the canopy CG.

#table(
  columns: (1fr, auto, auto, auto),
  stroke: 0.4pt + luma(170),
  inset: 5pt,
  align: (left, right, right, right),
  table.header([], [*Paper*], [*Robot*], [*ratio*]),

  table.cell(colspan: 4, fill: luma(235))[*Class I — dimensional. Read the robot column; ignore the paper column.*],
  [$m$ #h(0.5em) [kg]], [1.55], [0.3254], [0.210],
  [$b$ #h(0.5em) [m]], [2.15], [1.000], [0.465],
  [$overline(c)$ #h(0.5em) [m]], [0.54], [0.1961], [0.363],
  [$A^P$ #h(0.5em) [m²]], [1.16], [0.1961], [0.169],
  [$I_(x x)$ #h(0.5em) [kg·m²]], [0.336], [0.02002], [0.060],
  [$I_(y y)$], [0.292], [0.01622], [0.056],
  [$I_(z z)$], [0.109], [0.005290], [0.049],
  [$I_(x z)$], [$-0.059$], [$-9.48 times 10^(-5)$], [*0.002*],
  [$ell_P$ (canopy CG above sys CG) [m]], [1.066], [0.4152], [0.390],
  [$ell_F$ (pod CG below sys CG) [m]], [0.149], [0.1097], [0.736],

  table.cell(colspan: 4, fill: luma(235))[*Class II — mass distribution, $i = I slash (m b^2)$. These* should *match if the vehicles were similar.*],
  [$i_(x x)$], [0.04690], [0.06153], [1.312],
  [$i_(y y)$], [0.04075], [0.04983], [1.223],
  [$i_(z z)$], [0.01521], [0.01626], [1.069],
  [$i_(x z)$], [$-0.008235$], [$-0.000291$], [*0.035*],
  [$ell_P slash b$], [0.4958], [0.4152], [0.837],
  [$ell_F slash b$], [0.06930], [0.1097], [1.583],
  [Aspect ratio $b^2 slash A^P$], [3.985], [5.100], [1.280],

  table.cell(colspan: 4, fill: luma(235))[*Class III — the groups that decide whether identical coefficients give identical dynamics*],
  [Wing loading $m slash A^P$ #h(0.5em) [kg/m²]], [1.336], [1.660], [1.242],
  [Relative density $mu = m slash (rho A^P b)$], [0.5073], [1.355], [*2.670*],
  [$V_"trim"$ at $C_L = 0.6$ #h(0.5em) [m/s]], [5.97], [6.66], [1.114],
  [Froude $V^2 slash (g b)$], [1.691], [4.516], [*2.670*],
  [Reynolds ($overline(c)$, $V_"trim"$)], [$2.15 times 10^5$], [$0.87 times 10^5$], [*0.405*],
  [Time unit $t^* = b slash 2V$ #h(0.5em) [s]], [0.180], [0.0751], [0.417],
)

== Reading the audit

*The three bolded rows are the whole story.*

*$I_(x z)$ is not a small number that got smaller — it is absent.* The paper's
$i_(x z) = -0.0082$ is 18% of $i_(x x)$: that is a pilot-carrying trike with a
motor and cage slung well aft and below. The robot's $i_(x z) = -0.00029$ is
0.5% of $i_(x x)$ — it is very nearly symmetric about its $x$–$y$ plane, because
it is a symmetric box with a pusher motor on the centreline. *Transcribing the
paper's $I_(x z)$, at any scale factor, injects a roll–yaw inertial coupling
this vehicle does not have.* This is the single most damaging number to copy.

*$mu = 2.67 times$ and $"Fr" = 2.67 times$ are the same fact.* The robot is
2.67× denser relative to the air it flies in. Consequence, with an *identical*
$C_(l_p) = -0.127$: the non-dimensional roll subsidence eigenvalue is

$ hat(lambda) = C_(l_p) / (8 mu thin i_(x x)), $ <rollmode>

giving $hat(lambda) = -0.667$ for the paper and $-0.190$ for the robot — *3.5×
weaker damping in non-dimensional time.* Dimensionally the gap narrows, because
the small vehicle's clock runs 2.4× faster:

#table(columns: (auto, auto, auto, auto), stroke: 0.4pt + luma(170), inset: 5.5pt,
  align: (left, right, right, right),
  table.header([Roll subsidence, same $C_(l_p)$], [$hat(lambda)$], [$lambda$ [1/s]], [$tau$ [s]]),
  [Paper], [$-0.667$], [$-3.71$], [0.270],
  [Robot], [$-0.190$], [$-2.54$], [0.394],
)

So the *dimensional* roll time constant is only 1.46× slower — but you reach
that conclusion only by carrying $mu$ and $i_(x x)$ through. Anyone who
transplants the coefficients and assumes the dynamics come with them will be
wrong by 3.5× on every non-dimensional stability margin in the paper's §4.

*Re is 2.5× lower.* $0.87 times 10^5$ on a 125 µm single skin is a regime where
$C_(D_0)$ is dominated by laminar separation and is materially worse than at
$2.15 times 10^5$. $C_(D_0)^P = 0.15$ is the least defensible transfer in the set.

*Two ratios that are closer than expected, and are good news.* Wing loading
agrees to 24% and trim speed to 11%, so 6.0 m/s (`V_DESIGN`) remains the right
linearisation airspeed. And $ell_P slash b$ agrees to 16% — the robot's pendulum
geometry is a reasonable facsimile.

#note[
*A correction to `MODEL_NOTES.md` and to the previous revision of this plan.*
`LINE_HEIGHT_FRAC = 0.63` is the canopy *body origin* above the *pod origin*, and
comparing it to the paper's $1.066 slash 2.15 = 0.496$ is an apples-to-oranges
comparison — the paper's figure is canopy CG above *system* CG. The arched panels
hang below the canopy origin and the system CG sits 0.126 m above the pod origin,
so the true robot figure is $0.4152 slash 1.0 = 0.415$. The two rigs are 16% apart,
not 27%, and in the opposite direction to what a naive reading suggests.
]

= What actually has to change in the MJCF

The functional gap is not a matter of coefficients at all. MuJoCo's ellipsoid
fluid primitive (`fluidshape="ellipsoid"`, `fluidcoef="0.1 0.25 0.5 1.0 1.0"`)
has no $C_(L_0)$, no $C_(L_alpha)$, and no stability derivatives. *It is a
different function, not a different parameterisation of the same function.* No
value of `CANOPY_FLUIDCOEF` reproduces eq. (12). The paper's model has to be
added as an explicit force callback, and MuJoCo's own fluid model switched off.

== Disable the built-in fluid model — or double-count everything

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(170), inset: 5.5pt,
  table.header([*Mechanism*], [*Where*], [*Action*]),
  [Ellipsoid fluid on panels], [`class="skin"`], [delete `fluidshape`, `fluidcoef`],
  [Legacy inertia-box drag], [implied by `<option density viscosity>` on every geom without `fluidshape`], [set both to `0`],
  [Prop-disk opt-out hack], [`fluidcoef="0 0 0 0 0"` on `prop_disk`], [redundant once density is 0; remove],
)

Carry $rho = 1.225$ as a constant *in the callback*, not in `<option>`. Air
density then becomes a parameter of the aero model, which is also what lets you
sweep altitude later.

== Do not touch the inertia

The previous revision of this plan proposed emitting an explicit `<inertial>`
with the paper's tensor geometrically scaled. *That was wrong and should not be
done.* Against the compiled model it is off by:

#table(columns: (auto, auto, auto, auto), stroke: 0.4pt + luma(170), inset: 5.5pt,
  align: (left, right, right, right),
  table.header([], [Geometric scaling\ $k_m k_L^2 = 0.0454$], [Compiled model], [error]),
  [$I_(x x)$], [0.01526], [0.02002], [$-24%$],
  [$I_(y y)$], [0.01326], [0.01622], [$-18%$],
  [$I_(z z)$], [0.004950], [0.005290], [$-6%$],
  [$I_(x z)$], [$+0.002680$], [$-0.0000948$], [*28× and wrong sign*],
)

The robot's inertia is the accumulation of a traceable bill of materials. That
is *better* information than any scaling of another aircraft's tensor, and the
only correct action is to leave MuJoCo to compute it. The paper's tensor is used
for exactly one thing: the replication check of §6.

== Implement the aero layer in the paper's frame, and convert once

The paper is forward–right–down; the MJCF is forward–left–up. The previous
revision proposed a table of sign-flipped coefficients to type in. *Do not do
this either* — it is where I got $C_(l_(delta_a))$ and $C_(n_(delta_a))$ backwards,
and it is where anyone else will too.

Instead, convert the *state* in and the *force* out, and keep eqs. (8)–(22)
verbatim in between:

$ T = mat(1,0,0; 0,-1,0; 0,0,-1), quad
  v^"FRD" = T v^"FLU", quad
  f^"FLU" = T f^"FRD", quad
  M^"FLU" = T M^"FRD" $ <conv>

$T = T^(-1) = T^sans(T)$, so the same matrix serves in both directions and there
is one line of code to get right instead of fifteen table entries.

For cross-checking only, the correct signs are below. Note that $delta_L$ and
$delta_R$ name *physical* brake lines, so $delta_a = delta_L - delta_R$ is a
frame-independent scalar and does *not* flip — which is what makes
$C_(l_(delta_a))$ invariant and $C_(n_(delta_a))$ flip, not the other way round.

#table(
  columns: (auto, auto, 1fr),
  stroke: 0.4pt + luma(170), inset: 5.5pt,
  align: (left, center, left),
  table.header([*Quantity*], [*FRD → FLU*], [*Reason*]),
  [$L, p, phi$], [invariant], [$+$roll is right-wing-down in *both* frames],
  [$M, q, theta, alpha, beta$], [flip], [$y$ and $z$ reverse],
  [$N, r, psi$], [flip], [],
  [$delta_a, delta_s$], [invariant], [named by physical line, not by axis],
  [$C_(l_p), C_(l_phi), C_(m_q), C_(m_alpha), C_(n_r)$], [invariant], [$s_i s_j = +1$],
  [$C_(m_0)$], [flip], [$s_M = -1$, unpaired],
  [$C_(l_(delta_a))$], [*invariant*], [$s_L s_(delta_a) = +1$],
  [$C_(n_(delta_a))$], [*flip*], [$s_N s_(delta_a) = -1$],
  [$I_(x z)$], [flip], [$I' = T I T^sans(T)$],
)

== The remaining structural additions

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(170), inset: 5.5pt,
  table.header([*Term*], [*Equation, and what is missing today*], [*Status*]),
  [Parafoil $L$/$D$], [(12)–(16) at $P$, offset $[s_(P B)]$ from the CG], [replaces ellipsoid fluid],
  [Fuselage drag], [(8)–(11) at $F$. The pod has *zero* aero model today —
    `MODEL_NOTES.md` §10 records this.], [new],
  [Pure moments], [(18): $C_(l_p), C_(l_phi), C_(m_q), C_(m_0), C_(m_alpha), C_(n_r)$.
    The only rotational damping today is `fluidcoef[2] = 0.5`, a blunt-body term,
    not a stability derivative.], [new],
  [Brake force + moment], [(20)–(22), driven by a normalised $delta$ (§4.6)], [new],
  [$alpha$ clamp], [*not in the paper.* (15) is linear with no stall; at
    $C_(L_alpha) = 2$ the model reaches $C_L = 1.4$ by $30°$ and keeps going.], [new, ours],
)

*Correction, verified against MuJoCo rather than assumed
(`test_aero.py::test_xfrc_acts_at_com`):* `xfrc_applied` acts at the body
*centre of mass*, not at the body frame origin. Since the paper puts the
parafoil aerodynamic centre at the parafoil mass centre and the fuselage drag at
the fuselage mass centre, both forces land exactly where MuJoCo applies them.
*No offset moment is required at all* in the two-body model, and the leverage
moments of eq. (17) emerge through the tendons instead of being imposed. Only
the pure moments of eq. (18) are applied explicitly.

== Brakes: the input is not the same input

The MJCF actuates brakes geometrically — a 40 mm servo arm swinging through
3.0 rad, pulling a tendon anchored at the tip-panel trailing edge. The paper
uses $delta_(L slash R) in [0,1]$. The bridge is a *calibration*, not a physical
identity:

$ delta = "clamp"((ell(theta_"arm") - ell_0) / (ell_"max" - ell_0), 0, 1),
  quad ell(theta) = sqrt(D^2 + r^2 - 2 D r cos theta) $ <dnorm>

with $r =$ `ARM_LEN` $= 0.040$ m, $D$ the pivot-to-anchor distance, $ell_0$ the
rigged length including `BRAKE_FREE = 0.005` m. The generator already computes
@dnorm for the tendon `range`; it only has to be exposed as a scalar.

#warn[
*The brake coefficients cannot be transferred under any scaling.* Table 1 gives
$C_(l_(delta_a)) = +0.0021$; §3.2 identifies $C_(l_(delta_a)) = -0.2959$ — a
factor of 140 *and a sign flip*, on the same aircraft, in the same paper. The
table's nominal brake values are placeholders. The identified values are real,
but they were identified for a brake cascade of length $d = 0.40$ m on a 2.15 m
span ($d slash b = 0.186$) spreading load across the trailing edge. This robot
pulls *one tendon at one corner of one tip panel*. The authority distribution is
not similar, so neither is the coefficient.

Transfer the *structure* of (20)–(22) and the *sign pattern*; identify
$C_(l_(delta_a))$ and $C_(n_(delta_a))$ on this vehicle.
]

== Actuator lags: use the robot's own hardware, not the paper's

The paper's lags are its own hardware and are not vehicle-independent:

#table(columns: (auto, auto, auto, auto), stroke: 0.4pt + luma(170), inset: 5.5pt,
  align: (left, center, center, left),
  table.header([], [*Paper*], [*This robot*], [*Source*]),
  [Servo transit], [0.15 s/60°], [*0.09 s/60°* @ 7.4 V], [BD10BL-CAN datasheet, `itemized_spec.md` L123],
  [$omega_"servo" = 2.2 slash T_"rise"$], [14.7 rad/s], [*24.4 rad/s*], [],
  [$tau_"servo"$], [0.0680 s], [*0.0409 s*], [],
  [$tau_"motor"$], [0.4545 s], [*identify*], [their prop+ESC; ours is far lighter],
)

The robot's servo is 1.7× faster. Its rotor inertia is
$I_"spin" = 2.58 times 10^(-5)$ kg·m², which will give a spin-up constant far
below 0.45 s — take it from a bench step, do not copy 2.2 rad/s. Both go in as
native first-order activation dynamics:

```xml
<position class="servo_pos" name="servo_pos_L" joint="arm_L"
          dyntype="filter" dynprm="0.0409"/>
```

== Keep what the MJCF does better

The MJCF models propeller drag torque ($K_M slash K_T = 0.0105$ m, via the
`skydio_x2` `gear` trick) and carries a real `prop_spin` DOF so MuJoCo generates
the gyroscopic moment natively ($H approx 0.027$ N·m·s at full thrust). *The
paper has neither term.* On a 0.33 kg vehicle that pitches and yaws under brake,
these are the same order as the brake yaw moment. Keep them; document the
divergence from the paper rather than removing it to match.

= Implementation status

Sections 3.1, 3.3 and 3.4 are implemented and tested (32 assertions,
`test_aero.py`). Sections 3.5 (brakes) and 3.6 (actuator lags) are deliberately
untouched; `paramotor_aero.brake_wrench_frd()` is the hook and returns zero.

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(170), inset: 5.5pt,
  table.header([*Section*], [*Result*], [*State*]),
  [3.1], [`density="0" viscosity="0"`; no geom carries `fluidshape`/`fluidcoef`;
          `CANOPY_FLUIDCOEF` deleted. Vehicle falls at exactly
          $g dif t^2 n(n-1) slash 2$ with aero off — the discrete free-fall value,
          not $g t^2 slash 2$.], [done],
  [3.3], [Equations kept verbatim in FRD; $T = "diag"(1,-1,-1)$ applied once at
          the boundary. A guard refuses to run if MuJoCo's fluid model is
          re-enabled.], [done],
  [3.4], [Parafoil $L$/$D$, fuselage drag, six pure moments, $alpha$ clamp with
          excursion reporting.], [done],
  [3.5], [Brake hook present, returns zero.], [*not started*],
  [3.6], [Actuator lags absent.], [*not started*],
)

== What it does now

#table(
  columns: (1fr, auto, auto),
  stroke: 0.4pt + luma(170), inset: 5.5pt,
  align: (left, right, left),
  table.header([*Quantity*], [*Value*], [*Check*]),
  [Unpowered glide speed], [5.47 m/s], [audit predicts 6.7 at $C_L = 0.6$],
  [Sink rate], [1.69 m/s], [],
  [Glide ratio $L slash D$], [3.07], [`itemized_spec.md` assumes 2–3],
  [Powered climb at 1.5 N #super[†]], [$+0.79$ m/s], [],
  [Fuselage drag at 6 m/s], [0.340 N], [was *zero* before; 11% of weight],
  [Lift at $alpha = 0$, 6 m/s], [1.729 N], [$= 1/2 rho A^P V^2 C_(L_0)$ exactly],
  [Energy drift at $rho = 0$, 10 s], [$0.0$], [callback does no spurious work],
)

#super[†] with the propeller drag reaction removed — see below.

== Resolved: strip theory over the panels

The lumped single-force model could not produce a roll *restoring* moment, and
that — not insufficient damping — was the cause of the powered spiral. Damping
against a steady torque fixes the roll *rate*, so bank grows without bound; only
a restoring term bounds it. Strip theory applies eqs. (12)/(15) to each of the
fourteen panels separately, and because the canopy is arched each panel sees a
different local incidence, so the roll physics falls out of the geometry:

#table(
  columns: (1fr, auto, auto, auto),
  stroke: 0.4pt + luma(170), inset: 5.5pt, align: (left, right, right, left),
  table.header([], [*lumped*], [*strip*], [*note*]),
  [$C_(l_p)$], [$-0.127$], [$bold(-0.221)$], [1.74× the damping],
  [$C_(l beta)$], [0], [$bold(-0.292)$ /rad], [produced by *nothing* before],
  [$partial M slash partial beta$], [0], [$bold(-1.26)$ N·m/rad], [0.71° sideslip balances prop torque],
  [total lift], [1.729 N], [1.727 N], [preserved via arch recovery],
)

At 0.5 N the lumped model rolls to 63° and dives; strip holds *0.1°* and glides.
At 1.0 N it *climbs* with the propeller drag reaction left fully intact — the
lumped model needed that torque deleted to manage the same thing. The hardware
fact stays in the model, as §3.7 requires.

*Arch recovery.* An arched canopy lifts on its projected area, and strip theory
reproduces that exactly from geometry — measured 0.7979 against the generator's
`PROJ_FRACTION = 0.797`. But the paper's $C_(L_0)$ is referenced to *flat* area
on a wing that was already arched, so the loss is inside the coefficient.
Applying the arch again costs 20% of the lift and a 12%-high trim speed. The
factor (1.2533, computed from geometry) removes it from the coefficient so the
geometry can supply it.

#note[
*A correction to §4 of this plan.* $C_(l phi)$ was set to zero on the argument
that the suspension tendons produce the roll restoring moment structurally. They
do not — they produce the *gravity* pendulum, canopy displaced relative to pod.
The arc's *aerodynamic* dihedral effect is a separate mechanism and was modelled
by nothing at all. That was the direct cause of the spiral, and it is why the
lumped mode still spirals under power.
]

== Validated envelope, and one accepted limitation

#warn[
*The validated thrust range is 0 to about 1.0 N (T/W 0.31)*, and the spiral is
gone across all of it. Above that the vehicle pitches up, the tension-only
suspension goes fully slack (20 of 20 lines) and the canopy tumbles, with
$alpha$ reaching $plus.minus 180°$ — far outside the $-8°$/$+18°$ envelope where
a linear no-stall $C_L$ means anything.

Established *not* to be a thrust-line offset (moving the propeller from
$z = 0.010$ to the CG at $z = 0.126$ does not help) and *not* a step-input
artifact (ramping with the §3.6 motor lag $tau = 0.45$ s does not help). It needs
a stall model and a rigging re-trim, both existing plan items. The lumped model
was already failing above 0.5 N as a dive, so strip theory did not introduce
this — it widened the usable range.

*Accepted.* `test_aero.py::test_validated_thrust_envelope` pins both edges — 1.0 N
good, 1.8 N departs — so the boundary cannot drift unnoticed.
]

= Coefficient disposition

#table(
  columns: (auto, auto, auto, 1fr),
  stroke: 0.4pt + luma(170),
  inset: 5pt,
  align: (left, center, center, left),
  table.header([*Symbol*], [*Table 1*], [*Identified*], [*Disposition for this robot*]),

  table.cell(colspan: 4, fill: luma(235))[*Class II — transfer, with an AR correction from 3.99 → 5.10*],
  [$C_(L_0)^P$], [0.4], [—], [Use 0.4. Single-skin reflex camber; weakly AR-dependent.],
  [$C_(L_alpha)^P$], [2.0], [—], [*2.08.* Lifting line: invert at paper AR for
    $a_0 = 2.43$ ($e = 0.9$), re-evaluate at 5.10. A 4% change — defer if busy.],
  [$C_(D_alpha)^P$], [1.0], [—], [*0.92.* Induced part $C_(L_alpha)^2 slash (pi e "AR")$
    falls 0.355 → 0.277.],
  [$C_(l_p)$], [$-0.1$], [$-0.127$], [Use $-0.127$. Scales with $C_(L_alpha)$,
    which barely moves. *But see @rollmode*: same coefficient, 3.5× weaker
    non-dimensional damping.],
  [$C_(m_q)$], [$-2.0$], [—], [Use $-2.0$. Tail-less; set by $ell_P$, and
    $ell_P slash b$ agrees to 16%.],
  [$C_(n_r)$], [not tabulated], [$-0.0035$], [Use $-0.0035$. Note the paper omits
    it from Table 1 — a gap in the source.],
  [$chi$], [20°], [—], [Use 20° as the *initial* rigging angle, then trim it out.
    In the two-body MJCF this is emergent from line lengths, not a parameter.],

  table.cell(colspan: 4, fill: luma(235))[*Class III — do not transfer the value*],
  [$C_(D_0)^P$], [0.15], [—], [*Identify.* Re is 2.5× lower; also absorbs line drag,
    and this robot's line count and diameter differ.],
  [$C_(D_0)^F$, $C_(D_alpha)^F$, $A^F$], [0.15, 1.0, 0.5 m²], [—],
    [*Replace.* Only the product $C_(D_0)^F A^F$ is physical: the paper's is
     0.075 m² — a cage-and-pilot bluff body. The robot's pod is
     $0.140 times 0.110 = 0.0154$ m² frontal; use $A^F = 0.0154$, $C_(D_0)^F = 1.0$,
     giving 0.0154 m². Geometric scaling of $A^F$ (0.108 m²) is indefensible and
     was wrong in the previous revision.],
  [$C_(l_phi)$], [$-0.05$], [$-0.0055$], [*Zero in the two-body model.* This is the
    arc-plus-pendulum restoring moment; the suspension tendons already produce it
    structurally. Including it double-counts. Needed only in the rigid
    configuration of §6.],
  [$C_(m_0)$, $C_(m_alpha)$], [0.018, $-0.2$], [—], [*Re-trim.* Both are set by where
    the lines attach relative to the aerodynamic centre. The robot's A/B/C rows
    are at 0.060, $-0.060$, $-0.090$ in canopy $x$ — its own rigging, not the
    paper's. Use as a starting guess, then trim to $C_m = 0$ at $alpha_"trim"$.],
  [$C_(L_(delta_a))$, $C_(D_(delta_a))$], [0.0001, 0.0001], [—], [*Identify.* Both
    are $10^(-4)$ placeholders in Table 1 and are not credible as given.],
  [$C_(l_(delta_a))$], [0.0021], [$-0.2959$], [*Identify.* 140× disagreement within
    the paper; different brake cascade. Keep the *sign* as a sanity target.],
  [$C_(n_(delta_a))$], [0.004], [$-0.0506$], [*Identify.* Same. Flip sign on entry
    to FLU if not using @conv.],
  [$d$], [0.40 m], [—], [Set from this robot's own brake span. Only
    $C_(l_(delta_a)) b slash d$ is physical.],
)

= Validation: replicate first, then port

The two configurations must be kept distinct, and the replication is not
optional — it is the only way to know the aero layer is right before its outputs
become unfalsifiable.

*Configuration A — replication.* A throwaway MJCF of the *paper's* aircraft:
one rigid body, explicit `<inertial>` with $m = 1.55$ and
`fullinertia="0.336 0.292 0.109 0 0.059 0"` (note $I_(x z)$ flipped per @conv),
Table 1 verbatim, $C_(l_phi) = -0.0055$ present. Its only purpose is to
reproduce published figures.

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(170), inset: 5.5pt,
  table.header([*\#*], [*Test*], [*Reference*]),
  [A1], [Static, $alpha^P = 0$, $V_P = 6$: assert
        $L = 1/2 rho A^P V_P^2 C_(L_0) = 10.2$ N through $P$.], [eq. (12)],
  [A2], [Trim at throttle 0.54 of 10 N: level flight, $V approx 6$ m/s.], [§5, Fig. 7],
  [A3], [Throttle steps $+0.1 ... +0.3$, $-0.1$, $-0.2$: assert
        $dot(z) =$ 4.85, 4.83, 4.78, 4.75, 4.72 m/s, $K_"alt" approx 4.79$.], [Fig. 7],
  [A4], [Step $+0.2$, pitch response: 36% overshoot, $zeta = 0.31$,
        $omega_n = 1.53$ rad/s.], [Fig. 8, eq. (41)],
  [A5], [Numerically linearise about $theta = 0$; assert the state matrix matches
        eq. (25) elementwise.], [eq. (25)],
  [A6], [`ss2tf`; assert
        $T_"para" = (6.177 s^2 + 16.88 s + 47.11) slash (s^4 + 10.38 s^3 + 30.29 s^2 + 59.09 s)$.],
        [eq. (31)],
)

A6 is the strongest available check: it exercises the inertia tensor, all six
moment derivatives, the geometry vectors and @conv simultaneously, and a sign
error anywhere puts a pole in the wrong half-plane.

*Configuration B — the robot.* Same callback, robot parameters, MuJoCo-computed
inertia, two bodies, $C_(l_phi) = 0$.

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.4pt + luma(170), inset: 5.5pt,
  table.header([*\#*], [*Test*], [*Expected*]),
  [B1], [Energy: $rho = 0$, no thrust, 60 s — total energy conserved to $10^(-6)$.],
        [proves the callback does no spurious work],
  [B2], [Free trim: assert $V approx 6.7$ m/s at $C_L approx 0.6$.], [§2 audit],
  [B3], [Roll impulse: assert subsidence $tau approx 0.39$ s.], [@rollmode],
  [B4], [Pendulum: release from $10°$ bank, no aero. Assert $T approx 1.51$ s
        ($omega = 4.17$ rad/s).], [$sqrt(m g ell_P slash (I_(x x) + m ell_P^2))$],
  [B5], [Sign test: left brake only → *yaw left first*, roll following. Causal
        order matters — skid steering, not roll steering.], [§2.4],
  [B6], [$C_(l_phi)$ cross-check: run B rigid with $C_(l_phi) = -0.0055$ and
        two-body with $0$; assert the roll mode frequencies agree. Divergence
        means the tendon rigging is not producing the restoring moment.], [§4.x],
)

B4 is worth stating explicitly because it needs *no aerodynamics at all* and
depends only on mass, inertia and geometry — it will catch an inertia or CG
error before any coefficient is in play.

= Order of work

+ *Strip.* `density="0" viscosity="0"`, remove `fluidshape`/`fluidcoef`. Confirm
  the vehicle now falls ballistically — proof nothing else was quietly making lift.
+ *Write the callback* as a pure function of $("state", "params") arrow.r (f, M)$,
  in FRD, with @conv at the boundary. No MuJoCo imports beyond `mjData`.
+ *Configuration A*, Table 1 verbatim. Pass A1–A6. Do not proceed until A6 passes.
+ *Configuration B.* Swap the parameter dict, restore the two-body model, MuJoCo
  inertia, $C_(l_phi) = 0$. Pass B1–B6.
+ *Identify* $C_(D_0)^P$, $C_(m_0)$, $C_(m_alpha)$, $C_(l_(delta_a))$,
  $C_(n_(delta_a))$ on the real vehicle. The paper's §3.2 recursive weighted
  least-squares / Kalman method is directly reusable and is the most portable
  thing in the paper.

= Code layout

```
build_paramotor.py      - CANOPY_FLUIDCOEF;  density/viscosity -> 0
                          (inertia: NO CHANGE - leave MuJoCo to accumulate it)
paramotor_aero.py       NEW. Pure (state, params) -> (f, M) in FRD. Eqs. (8)-(22).
paramotor_params.py     NEW. PAPER_ACRA2012 and PEEK_1M dicts, identical keys.
check_paramotor.py      + tests A1-A6, B1-B6.
```

```python
T = np.diag([1.0, -1.0, -1.0])          # FLU <-> FRD, self-inverse

def aero(model, data):
    data.xfrc_applied[:] = 0.0
    R   = data.xmat[BODY].reshape(3, 3)
    v_b = T @ (R.T @ data.cvel[BODY, 3:])          # -> FRD body frame
    w_b = T @ (R.T @ data.cvel[BODY, :3])
    f, M = paramotor_aero.forces(v_b, w_b, phi, delta_L, delta_R, P)   # FRD
    data.xfrc_applied[BODY, :3] = R @ (T @ f)      # -> FLU -> world
    data.xfrc_applied[BODY, 3:] = R @ (T @ M)

mujoco.set_mjcb_passive(aero)
```

= Minimum change set

+ `<option density="0" viscosity="0">`; delete `fluidshape`/`fluidcoef` from
  `class="skin"` and `prop_disk`. Delete `CANOPY_FLUIDCOEF` — it has no image
  under this mapping.
+ Add `dyntype="filter"` with $tau_"servo" = 0.0409$ s from *this robot's*
  datasheet, not the paper's 0.0680 s. Identify $tau_"motor"$.
+ Add the `mjcb_passive` callback implementing eqs. (8)–(22) in FRD.
+ Introduce sixteen new coefficients — *seven* taken from the paper
  ($C_(L_0)^P$, $C_(L_alpha)^P$, $C_(D_alpha)^P$, $C_(l_p)$, $C_(m_q)$, $C_(n_r)$,
  $chi$), *nine* to be identified on this vehicle.
+ Take $A^P$, $b$, $c$, $[s_(P B)]$, $[s_(F B)]$, $[s_(M F)]$ and the full inertia
  from the compiled model. *Change none of them to match the paper.*
