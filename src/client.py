#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 + ZeroMQ client helpers.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import zmq


class MTYoloClient:
    """Thread-pool client with one REQ socket per worker call."""

    def __init__(self, servers: List[str], n_threads: int = 4, timeout_ms: int = 5000):
        if not servers:
            raise ValueError("servers list empty")
        self.servers = servers
        self.n_threads = n_threads
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self._lock = threading.Lock()
        self._idx = 0
        self.pool = ThreadPoolExecutor(max_workers=n_threads)

    def _new_socket(self):
        """Create a socket and connect to the next server in round-robin order."""
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        with self._lock:
            addr = self.servers[self._idx % len(self.servers)]
            self._idx += 1
        sock.connect(addr)
        return sock

    def send(self, *, payload: bytes, mode: int):
        """Send synchronously."""
        return self.pool.submit(self._worker, payload, mode).result()

    def _worker(self, payload: bytes, mode: int, opt=0):
        sock = self._new_socket()
        try:
            sock.send_multipart([payload, str(mode).encode(), str(opt).encode()])
            return sock.recv_json()
        finally:
            sock.close()

    def close(self):
        self.pool.shutdown(wait=True)

    def send_async(self, *, payload: bytes, mode: int, opt=0):
        """Send asynchronously and return a Future."""
        return self.pool.submit(self._worker, payload, mode, opt)

    def broadcast_cache(self, img_bytes: bytes):
        """Broadcast mode-1 cache request to all servers and wait for acknowledgments."""
        futures = []
        for addr in set(self.servers):
            fut = self.pool.submit(self._worker_to, addr, img_bytes, 1)
            futures.append((addr, fut))

        for addr, fut in futures:
            try:
                fut.result(timeout=self.timeout_ms / 1000 + 2)
            except Exception as exc:
                print(f"[WARN] broadcast to {addr} failed: {exc}")

    def _worker_to(self, addr: str, payload: bytes, mode: int):
        """Send one request to a specific server address."""
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(addr)
        try:
            sock.send_multipart([payload, str(mode).encode()])
            return sock.recv_json()
        finally:
            sock.close()
