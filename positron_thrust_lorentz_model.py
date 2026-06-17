"""
positron_thrust_lorentz_model.py

Antimatter Propulsion System Simulation
A 2D educational model of a proton/antiproton annihilation thruster,
where annihilation energy is converted into a directed pion stream
and focused into a thrust beam by a magnetic nozzle (Lorentz force).

Author:  Bolekbai Ulykpan (with AI from Claude & Gemini)
Course:  Mechanical / Aerospace Engineering Portfolio Project
Engine:  Python 3 + Pygame
"""

import pygame
import math
import random


# SECTION 1 — global configuration constants, tunable for experiments
# Window settings
WINDOW_WIDTH  = 1200
WINDOW_HEIGHT = 700
FPS           = 60
WINDOW_TITLE  = "Antimatter Propulsion System — Lorentz Magnetic Nozzle"

# Color palette (dark space theme)
COLOR_BG            = (6, 8, 16)          # deep space background
COLOR_STAR           = (255, 255, 255)
COLOR_INJECTOR_BODY = (90, 95, 110)
COLOR_INJECTOR_EDGE = (160, 170, 190)
COLOR_PROTON        = (255, 120, 60)      # protons rendered orange
COLOR_ANTIPROTON    = (90, 160, 255)      # antiprotons rendered blue
COLOR_PION_POS      = (255, 230, 80)      # positive pions — bright yellow
COLOR_PION_NEG      = (170, 80, 255)      # negative pions — violet
COLOR_FLASH         = (255, 255, 255)
COLOR_FIELD_LINE    = (60, 130, 200)
COLOR_TEXT          = (210, 220, 235)
COLOR_HUD_BG        = (10, 14, 24)
COLOR_HUD_BORDER    = (50, 90, 140)
COLOR_THRUST_BEAM   = (255, 180, 60)

# Physics constants (real-world values, used for E = mc^2 and cost calc)
SPEED_OF_LIGHT_MPS   = 299_792_458          # m/s, used in E = m * c^2
PROTON_MASS_KG       = 1.6726219e-27        # real proton rest mass
ANTIMATTER_COST_USD_PER_GRAM = 62_500_000_000_000  # $62.5 trillion per gram (illustrative)

# Simulation-scale physics constants (tuned for screen-space, not SI units)
PARTICLE_MASS_SIM     = 1.0      # simulation mass unit for injected protons/antiprotons
PION_MASS_SIM         = 0.3      # pions are lighter than protons -> deflect more easily
INJECTION_SPEED        = 3.0      # px/frame, how fast particles enter the mixing zone
PION_EJECTION_SPEED   = 6.0      # px/frame, pions are born with higher kinetic energy
B_FIELD_STRENGTH       = 0.12     # Tesla-equivalent, magnetic nozzle field strength
ANNIHILATION_RADIUS    = 14       # px, distance at which proton/antiproton annihilate
TIME_STEP              = 1.0      # simulation seconds per frame (Euler integration)

# Zone layout (x-coordinates divide the canvas into three stages)
INJECTOR_ZONE_X_END     = 260
MIXING_ZONE_X_START     = 260
MIXING_ZONE_X_END       = 620
NOZZLE_ZONE_X_START     = 620
NOZZLE_ZONE_X_END       = 1120

# Spawn control
SPAWN_INTERVAL_MS = 140    # milliseconds between new proton/antiproton pairs
MAX_REACTANTS     = 40     # cap on protons + antiprotons alive at once
MAX_PIONS         = 80     # cap on pions alive at once


# SECTION 2 — physics core, pure functions with no Pygame dependency

def lorentz_force_magnitude(charge, speed, b_field, alpha_rad):
    """
    Compute the magnitude of the Lorentz force acting on a moving charge.

    Formula:  F = q * v * B * sin(alpha)
        q     - electric charge of the particle (sim units)
        v     - particle speed (px/frame, simulation scale)
        B     - magnetic field strength (Tesla-equivalent)
        alpha - angle between the velocity vector and the B-field vector

    In this 2D model the magnetic field points out of the screen (+Z axis),
    so for any in-plane velocity the angle between v and B is 90 degrees,
    meaning sin(alpha) = 1. The function keeps alpha as a parameter so the
    model stays general and could later support a tilted field.
    """
    return charge * speed * b_field * math.sin(alpha_rad)


def apply_lorentz_deflection(vx, vy, charge, b_field, mass, dt):
    """
    Update a particle's 2D velocity under a magnetic field directed along +Z.

    Derivation:
        Lorentz force vector:  F = q * (v x B)
        With B = (0, 0, B_z), the cross product v x B gives:
            F_x =  q * vy * B_z
            F_y = -q * vx * B_z

    These components rotate the velocity vector over time without changing
    its magnitude (ideal magnetic confinement does no work on the charge).
    Newton's second law (a = F / m) is integrated using a simple Euler step.
    """
    force_x =  charge * vy * b_field
    force_y = -charge * vx * b_field

    accel_x = force_x / mass
    accel_y = force_y / mass

    new_vx = vx + accel_x * dt
    new_vy = vy + accel_y * dt
    return new_vx, new_vy


def annihilation_energy_joules(mass_kg_total):
    """
    Convert annihilated mass into released energy using Einstein's
    mass-energy equivalence formula:  E = m * c^2

    For a proton-antiproton pair, mass_kg_total should be the combined
    rest mass of both particles, since matter and antimatter mass are
    both fully converted to energy on annihilation.
    """
    return mass_kg_total * (SPEED_OF_LIGHT_MPS ** 2)


def estimate_thrust_newtons(pion_count_ejected, pion_speed_sim, dt_seconds):
    """
    Estimate instantaneous thrust force from momentum flux of ejected pions.

    Physics idea: Thrust = rate of momentum ejected per second (F = dp/dt).
    Since this is a 2D visual simulation (not a calibrated SI rocket model),
    we scale the simulation-space momentum by a fixed conversion factor to
    produce a plausible, smoothly varying Newton-scale readout for the HUD.
    This keeps the number physically motivated without claiming exact
    real-world thrust accuracy.
    """
    sim_momentum_per_frame = pion_count_ejected * PION_MASS_SIM * pion_speed_sim
    if dt_seconds <= 0:
        return 0.0
    # Conversion factor chosen so the HUD displays a readable, non-trivial
    # Newton-scale number for typical pion ejection rates in this model.
    SIM_TO_NEWTON_SCALE = 0.015
    return (sim_momentum_per_frame / dt_seconds) * SIM_TO_NEWTON_SCALE


def estimate_fuel_cost_usd(annihilated_pair_count):
    """
    Translate the number of annihilated proton-antiproton pairs into an
    illustrative dollar cost, based on the antimatter mass consumed and
    the assumed price of $62.5 trillion per gram.

    Note for the professor: this is a pedagogical cost illustration, not
    a real production-cost model — antimatter has never been manufactured
    at gram scale, so the constant is used purely to show how absurdly
    expensive even a few annihilation events would be.
    """
    total_mass_kg = annihilated_pair_count * (2 * PROTON_MASS_KG) * 1e17
    total_mass_g  = total_mass_kg * 1000.0
    return total_mass_g * ANTIMATTER_COST_USD_PER_GRAM


# SECTION 3 — particle classes

class ReactantParticle:
    """
    A proton or antiproton injected into the mixing zone.
    These particles travel in a straight line until they either annihilate
    with an opposite-type partner or are removed for going off-screen.
    """

    def __init__(self, is_antiproton):
        self.is_antiproton = is_antiproton
        self.mass = PARTICLE_MASS_SIM
        self.charge = -1.0 if is_antiproton else 1.0
        self.colour = COLOR_ANTIPROTON if is_antiproton else COLOR_PROTON

        # Antiprotons are injected from the top, protons from the bottom,
        # both drifting toward the mixing zone in the center of the canvas.
        if is_antiproton:
            self.x = float(random.randint(40, INJECTOR_ZONE_X_END - 40))
            self.y = 40.0
            self.vx = random.uniform(0.6, 1.4)
            self.vy = INJECTION_SPEED
        else:
            self.x = float(random.randint(40, INJECTOR_ZONE_X_END - 40))
            self.y = WINDOW_HEIGHT - 40.0
            self.vx = random.uniform(0.6, 1.4)
            self.vy = -INJECTION_SPEED

        self.alive = True
        self.annihilated = False

    def update(self, dt):
        """Advance position using simple linear motion (no field in this zone)."""
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Remove particles that drift off the visible canvas without colliding
        if (self.x < -20 or self.x > WINDOW_WIDTH + 20 or
                self.y < -20 or self.y > WINDOW_HEIGHT + 20):
            self.alive = False

    def draw(self, surface):
        pygame.draw.circle(surface, self.colour, (int(self.x), int(self.y)), 5)
        # Small glow ring to visually distinguish reactants from pions
        pygame.draw.circle(surface, self.colour, (int(self.x), int(self.y)), 8, 1)


class Pion:
    """
    A charged pion produced by proton-antiproton annihilation.
    Pions are the particles actually steered by the magnetic nozzle,
    since the Lorentz force only acts on moving charged particles.
    """

    TRAIL_LENGTH = 28

    def __init__(self, x, y, is_positive):
        self.is_positive = is_positive
        self.charge = 1.0 if is_positive else -1.0
        self.mass = PION_MASS_SIM
        self.colour = COLOR_PION_POS if is_positive else COLOR_PION_NEG

        self.x = x
        self.y = y

        # Pions are ejected toward the nozzle (rightward), with a small
        # random vertical spread representing the scatter angle from
        # the annihilation event.
        spread_angle = random.uniform(-0.35, 0.35)
        self.vx = PION_EJECTION_SPEED * math.cos(spread_angle)
        self.vy = PION_EJECTION_SPEED * math.sin(spread_angle)

        self.trail = []
        self.alive = True
        self.in_nozzle_field = False

    def update(self, dt, b_field):
        """
        Advance the pion's motion. If the pion has entered the magnetic
        nozzle zone, its velocity is updated using the Lorentz force model
        so the field bends its trajectory into the outgoing thrust beam.
        """
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > self.TRAIL_LENGTH:
            self.trail.pop(0)

        self.in_nozzle_field = NOZZLE_ZONE_X_START <= self.x <= NOZZLE_ZONE_X_END

        if self.in_nozzle_field:
            self.vx, self.vy = apply_lorentz_deflection(
                self.vx, self.vy, self.charge, b_field, self.mass, dt
            )

        self.x += self.vx * dt
        self.y += self.vy * dt

        if (self.x < -20 or self.x > WINDOW_WIDTH + 20 or
                self.y < -20 or self.y > WINDOW_HEIGHT + 20):
            self.alive = False

    def draw(self, surface):
        # Fading trail communicates trajectory curvature under the field
        trail_len = len(self.trail)
        for i, pos in enumerate(self.trail):
            t = i / max(trail_len - 1, 1)
            blend = 0.15 + t * 0.7
            r = int(COLOR_BG[0] * (1 - blend) + self.colour[0] * blend)
            g = int(COLOR_BG[1] * (1 - blend) + self.colour[1] * blend)
            b = int(COLOR_BG[2] * (1 - blend) + self.colour[2] * blend)
            radius = max(1, int(1 + t * 2))
            pygame.draw.circle(surface, (r, g, b), pos, radius)

        pygame.draw.circle(surface, self.colour, (int(self.x), int(self.y)), 3)


class AnnihilationFlash:
    """
    A short-lived bright flash drawn at the location of an annihilation
    event, representing the burst of energy released by E = m * c^2.
    Purely visual — it does not affect particle physics.
    """

    LIFETIME_FRAMES = 14

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.age = 0
        self.alive = True

    def update(self):
        self.age += 1
        if self.age >= self.LIFETIME_FRAMES:
            self.alive = False

    def draw(self, surface):
        progress = self.age / self.LIFETIME_FRAMES
        radius = int(4 + progress * 26)
        alpha = max(0, int(255 * (1 - progress)))

        flash_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(flash_surface, (*COLOR_FLASH, alpha), (radius, radius), radius)
        surface.blit(flash_surface, (int(self.x - radius), int(self.y - radius)))


# SECTION 4 — reactor manager, owns particle lists and HUD totals

class AntimatterReactor:
    """
    Central simulation manager: spawns reactant particles, detects
    annihilation events, spawns resulting pions, and accumulates the
    statistics displayed on the HUD.
    """

    def __init__(self):
        self.reactants = []
        self.pions = []
        self.flashes = []

        self._spawn_timer_ms = 0
        self.annihilation_count = 0
        self.total_energy_joules = 0.0
        self.current_thrust_n = 0.0
        self.total_fuel_cost_usd = 0.0

    def _spawn_reactant_pair(self):
        """Inject one proton and one antiproton together, as a matched pair."""
        if len(self.reactants) < MAX_REACTANTS:
            self.reactants.append(ReactantParticle(is_antiproton=True))
            self.reactants.append(ReactantParticle(is_antiproton=False))

    def _check_annihilations(self):
        """
        Compare every antiproton against every proton; if a pair is closer
        than ANNIHILATION_RADIUS, the pair is destroyed, energy is released,
        and two oppositely-charged pions are emitted toward the nozzle.
        """
        antiprotons = [p for p in self.reactants if p.is_antiproton and not p.annihilated]
        protons     = [p for p in self.reactants if not p.is_antiproton and not p.annihilated]

        for ap in antiprotons:
            for pr in protons:
                if pr.annihilated:
                    continue
                distance = math.hypot(ap.x - pr.x, ap.y - pr.y)
                if distance <= ANNIHILATION_RADIUS:
                    self._trigger_annihilation(ap, pr)
                    break  # this antiproton is consumed, move to the next one

    def _trigger_annihilation(self, antiproton, proton):
        """Handle the energy bookkeeping and particle creation for one event."""
        antiproton.annihilated = True
        proton.annihilated = True
        antiproton.alive = False
        proton.alive = False

        event_x = (antiproton.x + proton.x) / 2.0
        event_y = (antiproton.y + proton.y) / 2.0

        # Visual energy burst at the annihilation point
        self.flashes.append(AnnihilationFlash(event_x, event_y))

        # E = m * c^2 — both particles' rest mass is fully converted
        released_energy = annihilation_energy_joules(2 * PROTON_MASS_KG)
        self.total_energy_joules += released_energy

        # Real proton-antiproton annihilation typically yields several
        # pions; we model two for clarity — one positive, one negative —
        # so the magnetic nozzle visibly sorts them by charge sign.
        if len(self.pions) < MAX_PIONS:
            self.pions.append(Pion(event_x, event_y, is_positive=True))
        if len(self.pions) < MAX_PIONS:
            self.pions.append(Pion(event_x, event_y, is_positive=False))

        self.annihilation_count += 1
        self.total_fuel_cost_usd = estimate_fuel_cost_usd(self.annihilation_count)

    def update(self, dt, delta_ms, b_field):
        """Advance the whole reactor simulation by one frame."""

        # Inject new reactant pairs on a timer
        self._spawn_timer_ms += delta_ms
        if self._spawn_timer_ms >= SPAWN_INTERVAL_MS:
            self._spawn_reactant_pair()
            self._spawn_timer_ms = 0

        # Move existing particles
        for r in self.reactants:
            r.update(dt)
        for p in self.pions:
            p.update(dt, b_field)
        for f in self.flashes:
            f.update()

        # Detect new annihilation events this frame
        self._check_annihilations()

        # Remove dead/expired objects
        self.reactants = [r for r in self.reactants if r.alive]
        self.pions     = [p for p in self.pions if p.alive]
        self.flashes   = [f for f in self.flashes if f.alive]

        # Thrust is recomputed each frame from how many pions are
        # currently being accelerated through the nozzle field
        active_in_nozzle = sum(1 for p in self.pions if p.in_nozzle_field)
        self.current_thrust_n = estimate_thrust_newtons(
            active_in_nozzle, PION_EJECTION_SPEED, dt
        )

    def draw(self, surface):
        for r in self.reactants:
            r.draw(surface)
        for p in self.pions:
            p.draw(surface)
        for f in self.flashes:
            f.draw(surface)


# SECTION 5 — scene rendering (background, injector rig, magnetic nozzle)

def draw_starfield(surface, stars):
    """Render a static starfield for the deep-space background atmosphere."""
    for (sx, sy, brightness) in stars:
        shade = int(brightness)
        pygame.draw.circle(surface, (shade, shade, shade), (sx, sy), 1)


def generate_starfield(count):
    """Pre-compute star positions once at startup (avoids re-randomising every frame)."""
    stars = []
    for _ in range(count):
        x = random.randint(0, WINDOW_WIDTH)
        y = random.randint(0, WINDOW_HEIGHT)
        brightness = random.randint(60, 200)
        stars.append((x, y, brightness))
    return stars


def draw_injector_rig(surface):
    """
    Draw the particle source rig on the left side of the canvas.
    Styled as a 'Potassium-40 decay chamber' — a compact source producing
    a steady stream of protons (bottom) and antiprotons (top) for injection.
    """
    font_tiny = pygame.font.SysFont("consolas", 12)

    # Antiproton source housing (top-left)
    top_rect = pygame.Rect(30, 20, 160, 70)
    pygame.draw.rect(surface, COLOR_INJECTOR_BODY, top_rect, border_radius=6)
    pygame.draw.rect(surface, COLOR_ANTIPROTON, top_rect, 2, border_radius=6)
    label_top = font_tiny.render("ANTIPROTON SOURCE", True, COLOR_ANTIPROTON)
    surface.blit(label_top, (top_rect.x + 8, top_rect.y + 8))
    label_top2 = font_tiny.render("K-40 Decay Chamber", True, (150, 160, 180))
    surface.blit(label_top2, (top_rect.x + 8, top_rect.y + 28))

    # Proton source housing (bottom-left)
    bottom_rect = pygame.Rect(30, WINDOW_HEIGHT - 90, 160, 70)
    pygame.draw.rect(surface, COLOR_INJECTOR_BODY, bottom_rect, border_radius=6)
    pygame.draw.rect(surface, COLOR_PROTON, bottom_rect, 2, border_radius=6)
    label_bot = font_tiny.render("PROTON SOURCE", True, COLOR_PROTON)
    surface.blit(label_bot, (bottom_rect.x + 8, bottom_rect.y + 8))
    label_bot2 = font_tiny.render("K-40 Decay Chamber", True, (150, 160, 180))
    surface.blit(label_bot2, (bottom_rect.x + 8, bottom_rect.y + 28))

    # Beam guide rails — visually connect both sources toward the mixing zone
    pygame.draw.line(surface, COLOR_INJECTOR_EDGE,
                     (top_rect.centerx, top_rect.bottom),
                     (MIXING_ZONE_X_START, 100), 1)
    pygame.draw.line(surface, COLOR_INJECTOR_EDGE,
                     (bottom_rect.centerx, bottom_rect.top),
                     (MIXING_ZONE_X_START, WINDOW_HEIGHT - 100), 1)


def draw_mixing_zone(surface):
    """Draw a faint boundary marking the central annihilation/mixing zone."""
    zone_rect = pygame.Rect(MIXING_ZONE_X_START, 90,
                            MIXING_ZONE_X_END - MIXING_ZONE_X_START, WINDOW_HEIGHT - 180)
    overlay = pygame.Surface((zone_rect.width, zone_rect.height), pygame.SRCALPHA)
    overlay.fill((255, 255, 255, 8))
    surface.blit(overlay, (zone_rect.x, zone_rect.y))
    pygame.draw.rect(surface, (90, 90, 110), zone_rect, 1, border_radius=4)

    font_tiny = pygame.font.SysFont("consolas", 12)
    label = font_tiny.render("ANNIHILATION ZONE", True, (130, 130, 150))
    surface.blit(label, (zone_rect.centerx - 70, zone_rect.y + 6))


def draw_magnetic_nozzle(surface, b_field):
    """
    Draw the magnetic nozzle zone on the right side of the canvas using
    smooth curved field lines, representing the converging magnetic flux
    that focuses charged pions into a single directed thrust beam.
    """
    zone_rect = pygame.Rect(NOZZLE_ZONE_X_START, 60,
                            NOZZLE_ZONE_X_END - NOZZLE_ZONE_X_START, WINDOW_HEIGHT - 120)

    overlay = pygame.Surface((zone_rect.width, zone_rect.height), pygame.SRCALPHA)
    overlay.fill((30, 70, 120, 35))
    surface.blit(overlay, (zone_rect.x, zone_rect.y))
    pygame.draw.rect(surface, COLOR_HUD_BORDER, zone_rect, 2, border_radius=4)

    # Curved field lines converging toward the exit on the right —
    # purely illustrative of magnetic flux funneling the particle beam.
    center_y = WINDOW_HEIGHT // 2
    for offset in range(-240, 241, 60):
        start = (zone_rect.x, center_y + offset)
        control = (zone_rect.x + zone_rect.width * 0.5, center_y + offset * 0.4)
        end = (zone_rect.right, center_y + offset * 0.08)
        _draw_quadratic_curve(surface, start, control, end, COLOR_FIELD_LINE)

    font_tiny = pygame.font.SysFont("consolas", 12)
    label = font_tiny.render(f"MAGNETIC NOZZLE  ⊙  B = {b_field:.3f} T", True, COLOR_HUD_BORDER)
    surface.blit(label, (zone_rect.x + 10, zone_rect.y + 6))


def _draw_quadratic_curve(surface, start, control, end, colour, segments=24):
    """Helper: approximate a quadratic Bezier curve with line segments."""
    points = []
    for i in range(segments + 1):
        t = i / segments
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1]
        points.append((x, y))
    pygame.draw.lines(surface, colour, False, points, 1)


def draw_thrust_beam(surface, thrust_n):
    """
    Draw a glowing directional beam at the nozzle exit, whose brightness
    scales with the current estimated thrust, visualising the net force
    pushing the virtual spacecraft in the opposite direction.
    """
    exit_x = NOZZLE_ZONE_X_END
    exit_y = WINDOW_HEIGHT // 2
    intensity = min(1.0, thrust_n / 40.0)  # normalise for visual scaling only
    beam_length = int(40 + intensity * 60)
    beam_alpha = int(60 + intensity * 150)

    beam_surface = pygame.Surface((beam_length, 40), pygame.SRCALPHA)
    pygame.draw.polygon(
        beam_surface, (*COLOR_THRUST_BEAM, beam_alpha),
        [(0, 5), (0, 35), (beam_length, 20)]
    )
    surface.blit(beam_surface, (exit_x, exit_y - 20))


# SECTION 6 — HUD (heads-up display)

def draw_hud(surface, font, font_title, reactor, fps):
    """
    Render the real-time statistics panel: annihilation count, released
    energy (Joules), estimated thrust (Newtons), and illustrative fuel cost.
    """
    panel_rect = pygame.Rect(WINDOW_WIDTH - 300, 16, 284, 230)
    pygame.draw.rect(surface, COLOR_HUD_BG, panel_rect, border_radius=8)
    pygame.draw.rect(surface, COLOR_HUD_BORDER, panel_rect, 1, border_radius=8)

    title = font_title.render("REACTOR TELEMETRY", True, COLOR_HUD_BORDER)
    surface.blit(title, (panel_rect.x + 14, panel_rect.y + 10))
    pygame.draw.line(surface, COLOR_HUD_BORDER,
                     (panel_rect.x + 14, panel_rect.y + 32),
                     (panel_rect.right - 14, panel_rect.y + 32), 1)

    rows = [
        ("Annihilation events", f"{reactor.annihilation_count}"),
        ("Energy released (E=mc^2)", f"{reactor.total_energy_joules:.3e} J"),
        ("Estimated thrust", f"{reactor.current_thrust_n:.2f} N"),
        ("Fuel cost (illustrative)", f"${reactor.total_fuel_cost_usd:,.2f}"),
        ("Active reactants", f"{len(reactor.reactants)}"),
        ("Active pions", f"{len(reactor.pions)}"),
        ("Frame rate", f"{fps:.0f} FPS"),
    ]

    y = panel_rect.y + 44
    for label, value in rows:
        label_surf = font.render(label, True, (150, 160, 180))
        value_surf = font.render(value, True, COLOR_TEXT)
        surface.blit(label_surf, (panel_rect.x + 14, y))
        surface.blit(value_surf, (panel_rect.x + 14, y + 14))
        y += 30


def draw_legend(surface, font):
    """Small colour-coded legend so the particle types are unambiguous."""
    legend_rect = pygame.Rect(WINDOW_WIDTH - 300, 256, 284, 100)
    pygame.draw.rect(surface, COLOR_HUD_BG, legend_rect, border_radius=8)
    pygame.draw.rect(surface, COLOR_HUD_BORDER, legend_rect, 1, border_radius=8)

    entries = [
        (COLOR_PROTON, "Proton (p+)"),
        (COLOR_ANTIPROTON, "Antiproton (p-)"),
        (COLOR_PION_POS, "Positive pion (pi+)"),
        (COLOR_PION_NEG, "Negative pion (pi-)"),
    ]
    y = legend_rect.y + 12
    for colour, label in entries:
        pygame.draw.circle(surface, colour, (legend_rect.x + 16, y + 6), 5)
        text = font.render(label, True, COLOR_TEXT)
        surface.blit(text, (legend_rect.x + 32, y))
        y += 22


# SECTION 7 — main simulation loop

def main():
    """
    Entry point. Initialises Pygame, builds the reactor and scene objects,
    and runs the fixed-timestep loop until the window is closed.

    Extension points for future iterations:
        - Replace B_FIELD_STRENGTH with a live UI slider value.
        - Add a matplotlib/graph panel fed by reactor.total_energy_joules
          and reactor.current_thrust_n over time.
        - Add a pause/reset hotkey for classroom demonstrations.
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(WINDOW_TITLE)

    font       = pygame.font.SysFont("consolas", 13)
    font_title = pygame.font.SysFont("consolas", 14, bold=True)
    ticker     = pygame.time.Clock()

    stars   = generate_starfield(140)
    reactor = AntimatterReactor()
    b_field = B_FIELD_STRENGTH  # constant for now; swap for a slider value later

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        delta_ms = ticker.tick(FPS)

        reactor.update(TIME_STEP, delta_ms, b_field)

        # Render the full scene, back to front
        screen.fill(COLOR_BG)
        draw_starfield(screen, stars)
        draw_mixing_zone(screen)
        draw_magnetic_nozzle(screen, b_field)
        draw_injector_rig(screen)

        reactor.draw(screen)

        draw_thrust_beam(screen, reactor.current_thrust_n)
        draw_hud(screen, font, font_title, reactor, ticker.get_fps())
        draw_legend(screen, font)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()