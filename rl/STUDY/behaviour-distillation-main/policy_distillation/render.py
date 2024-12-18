import pygame
from pygame import gfxdraw
import numpy as np
from os import path

"""这个函数 render_cartpole 用于在一个 600x400 的屏幕上可视化强化学习中的倒立摆（CartPole）环境。
它接收当前状态 state 和环境参数 env_params，通过 Pygame 库绘制出一个模拟的倒立摆系统。
具体来说，函数绘制了一个带有小车和摆杆的二维图像，反映出小车的位置和摆杆的角度。
根据状态信息，计算小车和摆杆的位置、角度，并通过 Pygame 的绘图功能将它们显示出来，
最终返回一个处理后的图像数组，用于展示当前环境的状态。"""
def render_cartpole(state, env_params):
    screen_width = 600
    screen_height = 400
    length = 0.5
    x_threshold = 2.4

    pygame.init()
    screen = pygame.Surface((screen_width, screen_height))

    world_width = x_threshold * 2
    scale = screen_width / world_width
    polewidth = 10.0
    polelen = scale * (2 * length)
    cartwidth = 50.0
    cartheight = 30.0
    tau = env_params.tau

    if state is None:
        return None

    x = state

    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((255, 255, 255))

    for draw_mode in [0, 1]:

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0

        cartx = (x[0] + tau * x[1] * draw_mode) * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0, 250 - draw_mode * 150))
        gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0, 250 - draw_mode * 150))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-(x[2] + tau * x[3] * draw_mode))
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(surf, pole_coords, (202, 152, 101, 250 - draw_mode * 150))
        gfxdraw.filled_polygon(surf, pole_coords, (202, 152, 101, 250 - draw_mode * 150))

        gfxdraw.aacircle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203, 200 - draw_mode * 120),
        )
        gfxdraw.filled_circle(
            surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203, 200 - draw_mode * 120),
        )

        gfxdraw.hline(surf, 0, screen_width, carty, (0, 0, 0))

    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
    )


"""这个函数 render_acrobot 用于在一个 500x500 的屏幕上可视化强化学习中的双摆（Acrobot）环境。
它接收当前状态 state 和环境参数 env_params，并通过 Pygame 库绘制出一个包含两个连杆的双摆系统的二维图像。
具体来说，函数首先根据给定的角度（通过状态 state 中的角度信息）计算出两个连杆的位置，
然后在屏幕上绘制出连接这两个连杆的线条和圆点，模拟出双摆的动态状态。最终，返回处理过的图像数组，表示当前环境的状态。"""
def render_acrobot(state, env_params):
    from numpy import cos, pi, sin
    screen_dim = 500
    link_length_1 = 1.0
    link_length_2 = 1.0

    pygame.init()
    screen = pygame.Surface((screen_dim, screen_dim))

    surf = pygame.Surface((screen_dim, screen_dim))
    surf.fill((255, 255, 255))
    s = state

    bound = link_length_1 + link_length_2 + 0.2  # 2.2 for default
    scale = screen_dim / (bound * 2)
    offset = screen_dim / 2

    if s is None:
        return None

    p1 = [
        -link_length_1 * cos(s[0]) * scale,
        link_length_1 * sin(s[0]) * scale,
    ]

    p2 = [
        p1[0] - link_length_2 * cos(s[0] + s[1]) * scale,
        p1[1] + link_length_2 * sin(s[0] + s[1]) * scale,
    ]

    xys = np.array([[0, 0], p1, p2])[:, ::-1]
    thetas = [s[0] - pi / 2, s[0] + s[1] - pi / 2]
    link_lengths = [link_length_1 * scale, link_length_2 * scale]

    pygame.draw.line(
        surf,
        start_pos=(-2.2 * scale + offset, 1 * scale + offset),
        end_pos=(2.2 * scale + offset, 1 * scale + offset),
        color=(0, 0, 0),
    )

    for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
        x = x + offset
        y = y + offset
        l, r, t, b = 0, llen, 0.1 * scale, -0.1 * scale
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for coord in coords:
            coord = pygame.math.Vector2(coord).rotate_rad(th)
            coord = (coord[0] + x, coord[1] + y)
            transformed_coords.append(coord)
        gfxdraw.aapolygon(surf, transformed_coords, (0, 204, 204))
        gfxdraw.filled_polygon(surf, transformed_coords, (0, 204, 204))

        gfxdraw.aacircle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))
        gfxdraw.filled_circle(surf, int(x), int(y), int(0.1 * scale), (204, 204, 0))

    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
    )

"""这个函数 render_mountaincar 用于在一个 600x400 的屏幕上可视化强化学习中的山地车（MountainCar）环境。
它通过 Pygame 库绘制了一个包含山地车的环境，展示了山地车的当前位置、速度以及目标位置。

具体来说，函数首先绘制了一个由 sin 函数生成的山坡，代表山地车的行驶路径。
然后，它根据状态 state 中的位置（state[0]）来计算山地车的位置，并在屏幕上绘制出车身和车轮。
目标位置被标记为一个旗帜，显示在山坡的顶部。最后，函数返回一个处理过的图像数组，表示当前的环境状态。

函数的核心工作包括：

使用 sin 函数生成山坡的高度。
通过计算车身和车轮的位置，绘制出山地车。
在目标位置上绘制一个旗帜，标记山地车的目标。
返回渲染后的图像数据，便于后续的展示或处理。"""
def render_mountaincar(state, env_params):
    min_position = -1.2
    max_position = 0.6
    max_speed = 0.07
    goal_position = 0.5

    def _height(xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    force = 0.001
    gravity = 0.0025

    low = np.array([min_position, -max_speed], dtype=np.float32)
    high = np.array([max_position, max_speed], dtype=np.float32)

    screen_width = 600
    screen_height = 400

    pygame.init()
    screen = pygame.Surface((screen_width, screen_height))

    world_width = max_position - min_position
    scale = screen_width / world_width
    carwidth = 40
    carheight = 20

    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((255, 255, 255))

    pos = state[0]

    xs = np.linspace(min_position, max_position, 100)
    ys = _height(xs)
    xys = list(zip((xs - min_position) * scale, ys * scale))

    pygame.draw.aalines(surf, points=xys, closed=False, color=(0, 0, 0))

    clearance = 10

    l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
    coords = []
    for c in [(l, b), (l, t), (r, t), (r, b)]:
        c = pygame.math.Vector2(c).rotate_rad(np.cos(3 * pos))
        coords.append(
            (
                c[0] + (pos - min_position) * scale,
                c[1] + clearance + _height(pos) * scale,
            )
        )

    gfxdraw.aapolygon(surf, coords, (0, 0, 0))
    gfxdraw.filled_polygon(surf, coords, (0, 0, 0))

    for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
        c = pygame.math.Vector2(c).rotate_rad(np.cos(3 * pos))
        wheel = (
            int(c[0] + (pos - min_position) * scale),
            int(c[1] + clearance + _height(pos) * scale),
        )

        gfxdraw.aacircle(
            surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
        )
        gfxdraw.filled_circle(
            surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
        )

    flagx = int((goal_position - min_position) * scale)
    flagy1 = int(_height(goal_position) * scale)
    flagy2 = flagy1 + 50
    gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))

    gfxdraw.aapolygon(
        surf,
        [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
        (204, 204, 0),
    )
    gfxdraw.filled_polygon(
        surf,
        [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
        (204, 204, 0),
    )

    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
    )

"""
函数 render_fourrooms 用于可视化强化学习中的四房间环境。它通过 matplotlib 绘制一个 2D 环境地图，
标记出墙壁和可通行区域，并在图中用红色和绿色分别标示出当前代理的位置和目标位置。
该函数返回一个图形对象和坐标轴，方便进行环境的展示和进一步处理。"""
def render_fourrooms(state, env_params):
    """Small utility for plotting the agent's state."""
    import matplotlib.pyplot as plt
    import chex
    import jax.numpy as jnp

    four_rooms_map = """
xxxxxxxxxxxxx
x     x     x
x     x     x
x           x
x     x     x
x     x     x
xx xxxx     x
x     xxx xxx
x     x     x
x     x     x
x           x
x     x     x
xxxxxxxxxxxxx"""

    def _string_to_bool_map(str_map: str) -> chex.Array:
        """Convert string map into boolean walking map."""
        bool_map = []
        for row in str_map.split("\n")[1:]:
            bool_map.append([r == " " for r in row])
        return jnp.array(bool_map)

    env_map = _string_to_bool_map(four_rooms_map)
    occupied_map = 1 - env_map

    fig, ax = plt.subplots()
    ax.imshow(occupied_map, cmap="Greys")
    ax.annotate(
        "A",
        fontsize=20,
        xy=(state.pos[1], state.pos[0]),
        xycoords="data",
        xytext=(state.pos[1] - 0.3, state.pos[0] + 0.25),
        color="red"
    )
    ax.annotate(
        "G",
        fontsize=20,
        xy=(state.goal[1], state.goal[0]),
        xycoords="data",
        xytext=(state.goal[1] - 0.3, state.goal[0] + 0.25),
        color="green"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

"""这个函数 render_pendulum 用于在 500x500 的屏幕上可视化倒立摆环境。
它使用 Pygame 渲染倒立摆系统，显示摆杆的角度和当前的动作。
具体来说，函数根据给定的角度 state[0] 绘制摆杆，使用图标（如顺时针或逆时针）来可视化动作，
并根据动作的大小和方向调整图标的大小和方向，同时在摆杆的转轴位置绘制一个小圆点。最后，函数返回渲染后的图像数据。"""
def render_pendulum(state, action, env_params):

    # Render action
    last_u = action
    
    screen_dim = 500

    pygame.init()
    screen = pygame.Surface((screen_dim, screen_dim))

    surf = pygame.Surface((screen_dim, screen_dim))
    surf.fill((255, 255, 255))

    bound = 2.2
    scale = screen_dim / (bound * 2)
    offset = screen_dim // 2

    rod_length = 1 * scale
    rod_width = 0.2 * scale
    l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
    coords = [(l, b), (l, t), (r, t), (r, b)]
    transformed_coords = []
    for c in coords:
        c = pygame.math.Vector2(c).rotate_rad(state[0] + np.pi / 2)
        c = (c[0] + offset, c[1] + offset)
        transformed_coords.append(c)
    gfxdraw.aapolygon(surf, transformed_coords, (204, 77, 77))
    gfxdraw.filled_polygon(surf, transformed_coords, (204, 77, 77))

    gfxdraw.aacircle(surf, offset, offset, int(rod_width / 2), (204, 77, 77))
    gfxdraw.filled_circle(
        surf, offset, offset, int(rod_width / 2), (204, 77, 77)
    )

    rod_end = (rod_length, 0)
    rod_end = pygame.math.Vector2(rod_end).rotate_rad(state[0] + np.pi / 2)
    rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
    gfxdraw.aacircle(
        surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    )
    gfxdraw.filled_circle(
        surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
    )

    fname = path.join(path.dirname(__file__), "assets/clockwise.png")
    img = pygame.image.load(fname)
    if last_u is not None:
        scale_img = pygame.transform.smoothscale(
            img,
            (scale * np.abs(last_u) / 2, scale * np.abs(last_u) / 2),
        )
        is_flip = bool(last_u > 0)
        scale_img = pygame.transform.flip(scale_img, is_flip, True)
        surf.blit(
            scale_img,
            (
                offset - scale_img.get_rect().centerx,
                offset - scale_img.get_rect().centery,
            ),
        )

    # drawing axle
    gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
    gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))

    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))

    return np.transpose(
        np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
    )