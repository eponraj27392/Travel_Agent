import asyncio
from dotenv import load_dotenv; load_dotenv()
from tourist_agent.memory_store import init_store

async def check():
  store = await init_store()
  profile = await store.aget(('user_profile', 'user-esakki-id'), 'profile')
  print('Profile:', profile.value if profile else 'not yet saved')

asyncio.run(check())