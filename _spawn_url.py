import asyncio
from menlo_runner.config import load_config
from menlo_runner.context import RobotContext

async def main():
    config = load_config(require_tokamak=False)
    ctx = await RobotContext.create(config, name_prefix="test-scenario")
    print("ROBOT_ID=", ctx.robot_id)
    print("VIEWER_URL=", ctx.viewer_url)
    await ctx.close()

asyncio.run(main())
