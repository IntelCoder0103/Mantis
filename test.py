import asyncio, bittensor as bt, requests, config, comms, logging, os
from timelock import Timelock
import json
import aiohttp
import logging

import ast

logger = logging.getLogger(__name__)

MAX_PAYLOAD_BYTES = 25 * 1024 * 1024

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)

async def _get_drand_signature(round_num: int) -> bytes | None:
  await asyncio.sleep(2)
  try:
      url = f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/rounds/{round_num}"
      async with aiohttp.ClientSession() as session:
          async with session.get(url, timeout=10) as response:
              if response.status == 200:
                  data = await response.json()
                  logger.debug(f"✅ Signature fetched for round {round_num}")
                  return bytes.fromhex(data["signature"])
              else:
                  logger.warning(
                      f"-> Failed to fetch signature for round {round_num}, "
                      f"status: {response.status}"
                  )
                  return None
  except asyncio.TimeoutError:
      logger.warning(f"-> Timeout fetching signature for round {round_num}")
      return None
  except Exception as e:
      logger.error(f"-> Error fetching signature for round {round_num}: {e}")
      return None

async def main():
  object_url = "https://pub-e508ff3c583c4237989a125d2f2db35b.r2.dev/5DJnNPMgkVEQZ2URiJPaQejK4rAw1D8koLt5VTdvbbFDrTHy"
  payload_raw = await comms.download(object_url, max_size_bytes=MAX_PAYLOAD_BYTES)
  
  print(payload_raw)
  
  payload = payload_raw
  # payload = json.loads(str(payload_raw))
  ct_hex = payload["ciphertext"]
  round_num = payload["round"]
  tlock = Timelock(DRAND_PUBLIC_KEY)
  sig = await _get_drand_signature(round_num)
  
  pt_bytes = tlock.tld(bytes.fromhex(ct_hex), sig)
  
  DECRYPTED_PAYLOAD_LIMIT_BYTES = 32 * 1024 # 32KB limit
  if len(pt_bytes) > DECRYPTED_PAYLOAD_LIMIT_BYTES:
      raise ValueError(f"Decrypted payload size {len(pt_bytes)} exceeds limit")

  full_plaintext = pt_bytes.decode('utf-8')
  delimiter = ":::"
  parts = full_plaintext.rsplit(delimiter, 1)
  embeddings_str, payload_hotkey = parts
  submission = ast.literal_eval(embeddings_str)

  print(submission)

asyncio.run(main())