import asyncio
import time
from menlo_runner.config import load_config
from menlo_runner.context import RobotContext

async def main():
    config = load_config(require_tokamak=False)
    ctx = await RobotContext.create(config, name_prefix="test-scenario")
    print("ROBOT_ID=", ctx.robot_id)
    print("VIEWER_URL=", ctx.viewer_url)
    print("Waiting for viewer to connect (skills advertise)...")
    skills = await ctx.wait_for_skills(timeout_s=180)
    print(f"Found {len(skills)} skills:", [s.name for s in skills])

    # short movement test
    print("--- Movement test ---")
    r1 = await ctx.invoke("set_head", {"yaw": 0.0, "pitch": 0.15}, timeout_s=10)
    print("set_head:", r1)
    await asyncio.sleep(0.5)
    r2 = await ctx.invoke("set_velocity", {"vx": 0.5, "wz": 0.0, "duration_s": 1.0}, timeout_s=15)
    print("set_velocity fwd:", r2)
    print("--- Test done. Cleaning up. ---")
    await ctx.close()

asyncio.run(main())
