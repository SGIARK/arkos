import asyncio
from config_module.loader import config
from tool_module.tool_call import MCPToolManager

async def test():
    manager = MCPToolManager(config.get('mcp_servers'))
    await manager.initialize_servers()
    tools = await manager.list_all_tools()
    print('Tools:', tools)
    await manager.shutdown()

asyncio.run(test())
