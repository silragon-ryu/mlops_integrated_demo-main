import pytest
import sys
import os

# Add project root to path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import stable_hash_to_bucket

# --- PART 1: UNIT TEST (Isolated Logic) ---

def test_hashing_determinism():
    """
    Requirement: Verify logic consistency.
    The same input must ALWAYS produce the same bucket index.
    """
    input_val = "item_12345"
    bucket_1 = stable_hash_to_bucket(input_val, num_buckets=100)
    bucket_2 = stable_hash_to_bucket(input_val, num_buckets=100)
    
    assert bucket_1 == bucket_2
    assert isinstance(bucket_1, int)

def test_hashing_range():
    """
    Requirement: Verify logic boundary conditions.
    Output must be strictly between 0 and num_buckets - 1.
    """
    buckets = 50
    # Test multiple inputs to ensure none go out of bounds
    for i in range(100):
        val = f"test_val_{i}"
        result = stable_hash_to_bucket(val, buckets)
        assert 0 <= result < buckets

def test_hashing_invalid_input():
    """
    Requirement: Verify error handling.
    """
    with pytest.raises(ValueError):
        # Your code explicitly raises ValueError if buckets <= 0
        stable_hash_to_bucket("test", num_buckets=0)