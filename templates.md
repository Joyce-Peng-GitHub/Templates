# Templates

[TOC]

<div STYLE="page-break-after: always;"></div>

## Utilities

### Iterator Requirements

```cpp
template <typename Iter>
using RequireInputIter = typename std::enable_if<
	std::is_base_of<
		std::input_iterator_tag,
		typename std::iterator_traits<Iter>::iterator_category>::value>::type;

template <typename Iter>
using RequireFwdIter = typename std::enable_if<
	std::is_base_of<
		std::forward_iterator_tag,
		typename std::iterator_traits<Iter>::iterator_category>::value>::type;

template <typename Iter>
using RequireBidirIter = typename std::enable_if<
	std::is_base_of<
		std::bidirectional_iterator_tag,
		typename std::iterator_traits<Iter>::iterator_category>::value>::type;

template <typename Iter>
using RequireRAIter = typename std::enable_if<
	std::is_base_of<
		std::random_access_iterator_tag,
		typename std::iterator_traits<Iter>::iterator_category>::value>::type;
```

### Register `__uint128_t` as Unsigned

```cpp
namespace std {
	template <>
	struct is_unsigned<__uint128_t> : public true_type {};
}
```

### $\overset{\min}{\gets}$ and $\overset{\min}{\gets}$

```cpp
template <typename T, typename Cmp = std::less<T>>
inline T &minEq(T &lhs, const T &rhs, Cmp cmp = Cmp()) {
	return (cmp(rhs, lhs) ? (lhs = rhs) : lhs);
}

template <typename T, typename Cmp = std::less<T>>
inline T &maxEq(T &lhs, const T &rhs, Cmp cmp = Cmp()) {
	return (cmp(lhs, rhs) ? (lhs = rhs) : lhs);
}
```

### Binary Search

```cpp
/**
 * @brief behaves as if it generates func[beg, end) and
 * 	performs std::lower_bound on it
 */
template <typename T, typename Ret, typename Func,
		  typename Cmp = std::less<Ret>>
inline T lowerBound(T beg, T end, const Ret &val, Func func = Func(), Cmp cmp = Cmp()) {
	while (beg < end) {
		T mid = beg + (end - beg) / 2;
		if (cmp(func(mid), val)) {
			beg = mid + 1;
		} else {
			end = mid;
		}
	}
	return beg;
}

/**
 * @brief behaves as if it generates func[beg, end) and
 * 	performs std::upper_bound on it
 */
template <typename T, typename Ret, typename Func,
		  typename Cmp = std::less<Ret>>
inline T upperBound(T beg, T end, const Ret &val, Func func = Func(), Cmp cmp = Cmp()) {
	while (beg < end) {
		T mid = beg + (end - beg) / 2;
		if (cmp(val, func(mid))) {
			end = mid;
		} else {
			beg = mid + 1;
		}
	}
	return beg;
}
```

### $\log_2$ of Integers

```cpp
inline unsigned floorLogn2(uint64_t x) {
#if __cplusplus >= 202002L
	return (std::bit_width(x) - 1);
#else
	return (x ? (sizeof(uint64_t) * 8 - __builtin_clzll(x) - 1) : -1);
#endif
}

inline unsigned ceilLogn2(uint64_t x) {
	return (x ?
#if __cplusplus >= 202002L
			  std::bit_width(x - 1)
#else
			  ((x == 1) ? 0 : (sizeof(uint64_t) * 8 - __builtin_clzll(x - 1)))
#endif
			  : -1);
}
```

<div STYLE="page-break-after: always;"></div>

## Optimization Tricks

### Compiler Optimization Switch

```cpp
#pragma GCC optimize(2)					   // O2
#pragma GCC optimize(3, "Ofast", "inline") // O3
```

### Fast Modulo

```cpp
template <typename Mod = uint32_t,
		  typename Mu = uint64_t,
		  typename Aux = __uint128_t,
		  typename = std::enable_if_t<std::is_unsigned<Mod>::value &&
									  std::is_unsigned<Mu>::value &&
									  std::is_unsigned<Aux>::value &&
									  sizeof(Mod) * 2 == sizeof(Mu) &&
									  sizeof(Mu) * 2 == sizeof(Aux)>>
class Barrett {
public:
	explicit Barrett(Mod mod) : m_mod(mod), m_mu((Mu)(-1) / mod + 1) {}

	void assign(Mod mod) {
		m_mod = mod;
		m_mu = (Mu)(-1) / mod + 1;
	}

	Mod umod() const { return m_mod; }

	Mod operator()(Mu x) const {
		Mu p = ((static_cast<Aux>(x) * m_mu) >> (sizeof(Mu) * 8)) * m_mod;
		return ((x < p) ? (x - p + m_mod) : (x - p));
	}
	Mod operator()(Mod lhs, Mod rhs) const {
		return operator()(static_cast<Mu>(lhs) * rhs);
	}

protected:
	Mod m_mod;
	Mu m_mu;

private:
};
```

<div STYLE="page-break-after: always;"></div>

## Data Structures

### Disjoint Set Union

```cpp
class Dsu {
public:
	Dsu(size_t n = 0) : m_fa_or_sz(n, -1) {}

	void assign(size_t n) { m_fa_or_sz.assign(n, -1); }
	void clear() { m_fa_or_sz.clear(); }
	void resize(size_t n) { m_fa_or_sz.resize(n, -1); }

	size_t size() const noexcept { return m_fa_or_sz.size(); }
	bool empty() const noexcept { return m_fa_or_sz.empty(); }

	size_t leader(size_t x) {
		if (m_fa_or_sz[x] < 0) {
			return x;
		}
		return (m_fa_or_sz[x] = leader(m_fa_or_sz[x]));
	}
	bool same(size_t x, size_t y) { return (leader(x) == leader(y)); }

	size_t size(size_t x) { return -m_fa_or_sz[leader(x)]; }

	size_t merge(size_t x, size_t y) {
		x = leader(x);
		y = leader(y);
		if (x == y) {
			return x;
		}
		if (-m_fa_or_sz[x] < -m_fa_or_sz[y]) {
			std::swap(x, y);
		}
		m_fa_or_sz[x] += m_fa_or_sz[y];
		m_fa_or_sz[y] = x;
		return x;
	}

	std::vector<std::vector<size_t>> groups() {
		std::vector<size_t> leader_of(size()), sz(size());
		for (size_t i = 0; i != size(); ++i) {
			++sz[leader_of[i] = leader(i)];
		}
		std::vector<std::vector<size_t>> grps(size());
		for (size_t i = 0; i != size(); ++i) {
			grps[leader_of[i]].reserve(sz[leader_of[i]]);
		}
		for (size_t i = 0; i != size(); ++i) {
			grps[leader_of[i]].emplace_back(i);
		}
		grps.erase(std::remove_if(grps.begin(), grps.end(),
								  [](const std::vector<size_t> &grp) {
									  return grp.empty();
								  }),
				   grps.end());
		return grps;
	}

protected:
	std::vector<ssize_t> m_fa_or_sz; // if root, stores -size; else stores fa

private:
};
```

### Sparse Table

```cpp
template <typename T, typename Cmp = std::less<>>
struct Min {
	inline const T &operator()(const T &lhs, const T &rhs) const {
		return std::min(lhs, rhs, Cmp());
	}
};

template <typename T, typename Cmp = std::less<>>
struct Max {
	inline const T &operator()(const T &lhs, const T &rhs) const {
		return std::max(lhs, rhs, Cmp());
	}
};

template <typename _T, typename _Oper = Min<_T>>
class SparseTable {
public:
	using Oper = _Oper;
	using Elem = _T;
	using Table = std::vector<std::vector<Elem>>;

	inline static unsigned floorLogn2(uint64_t x) {
#if __cplusplus >= 202002L
		return (std::bit_width(x) - 1);
#else
		return (x ? (sizeof(uint64_t) * 8 - __builtin_clzll(x) - 1) : -1);
#endif
	}

	SparseTable(const Oper &oper = Oper()) : m_oper(oper) {}
	template <typename Iter, typename = RequireInputIter<Iter>>
	SparseTable(Iter begin, Iter end, const Oper &oper = Oper())
		: m_oper(oper), m_data(std::distance(begin, end)) {
		for (size_t i = 0; i != size(); ++i) {
			m_data[i].resize(floorLogn2(size() - i) + 1);
			m_data[i][0] = *(begin++);
		}
		for (size_t i = 1; (size_t(1) << i - 1) < size(); ++i) {
			for (size_t j = 0; j + (size_t(1) << i) <= size(); ++j) {
				m_data[j][i] = m_oper(m_data[j][i - 1],
									  m_data[j + (size_t(1) << (i - 1))][i - 1]);
			}
		}
	}

	inline void assign(const Oper &oper = Oper()) {
		m_oper = oper;
		m_data.clear();
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline void assign(Iter begin, Iter end, const Oper &oper = Oper()) {
		m_oper = oper;
		m_data.resize(std::distance(begin, end));
		for (size_t i = 0; i != size(); ++i) {
			m_data[i].resize(floorLogn2(size() - i) + 1);
			m_data[i][0] = *(begin++);
		}
		for (size_t i = 1; (size_t(1) << (i - 1)) < size(); ++i) {
			for (size_t j = 0; j + (size_t(1) << i) <= size(); ++j) {
				m_data[j][i] = m_oper(m_data[j][i - 1],
									  m_data[j + (size_t(1) << (i - 1))][i - 1]);
			}
		}
	}

	inline void clear() noexcept { m_data.clear(); }

	inline size_t size() const noexcept { return m_data.size(); }
	inline bool empty() const noexcept { return m_data.empty(); }

	inline Elem query(size_t pos, size_t len) const {
		if (pos >= m_data.size()) {
			throw std::out_of_range("SparseTable::query: pos (which is " +
									std::to_string(pos) +
									") > size() (which is " +
									std::to_string(m_data.size()) + ')');
		}
		if (len == 0) {
			throw std::out_of_range("SparseTable::query: len == 0");
		}
		if (pos + len > size()) {
			len = size() - pos;
		}
		size_t log_len = floorLogn2(len);
		return m_oper(m_data[pos][log_len],
					  m_data[pos + len - (size_t(1) << log_len)][log_len]);
	}

protected:
	Oper m_oper;
	Table m_data; // m_data[i][j] maintains Oper(data[i, i + 2 ** j))

private:
};
```

### Monotonic Deque

```cpp
template <typename T, typename Cmp = std::less<T>, typename Dq = std::deque<T>>
class MonoDeque {
public:
	inline MonoDeque(const Cmp &cmp = Cmp()) : m_cmp(cmp) {}

	inline size_t size() const noexcept { return m_dq.size(); }
	inline bool empty() const noexcept { return m_dq.empty(); }

	inline const T &front() const { return m_dq.front(); }
	inline const T &back() const { return m_dq.back(); }

	inline void pushFront(const T &val) {
		popFrontFor(val);
		m_dq.push_front(val);
	}
	inline void pushFront(T &&val) {
		popFrontFor(val);
		m_dq.push_front(std::move(val));
	}
	template <typename... Args>
	inline void emplaceFront(Args &&...args) {
		auto t = T(std::forward(args)...);
		popFrontFor(t);
		m_dq.push_front(std::move(t));
	}

	inline void pushBack(const T &val) {
		popBackFor(val);
		m_dq.push_back(val);
	}
	inline void pushBack(T &&val) {
		popBackFor(val);
		m_dq.push_back(std::move(val));
	}
	template <typename... Args>
	inline void emplaceBack(Args &&...args) {
		auto t = T(std::forward<Args>(args)...);
		popBackFor(t);
		m_dq.push_back(std::move(t));
	}

	inline void popFront() { m_dq.pop_front(); }
	inline void popFrontFor(const T &val) {
		while (m_dq.size() && !m_cmp(val, m_dq.front())) {
			popFront();
		}
	}

	inline void popBack() { m_dq.pop_back(); }
	inline void popBackFor(const T &val) {
		while (size() && !m_cmp(m_dq.back(), val)) {
			popBack();
		}
	}

	inline void clear() noexcept { m_dq.clear(); }

	inline void swap(MonoDeque &another) noexcept {
		std::swap(m_cmp, another.m_cmp);
		std::swap(m_dq, another.m_dq);
	}

protected:
	Cmp m_cmp;
	Dq m_dq;
};
```

### Fenwick Tree

```cpp
template <typename T, typename Oper = std::plus<T>, T id_elem = T()>
class FenwickTree {
public:
	inline static size_t lowbit(size_t x) { return (x & (-x)); }

	inline explicit FenwickTree(size_t n = 0) : m_tree(n + 1, id_elem) {}
	inline explicit FenwickTree(std::initializer_list<T> list) : m_tree(list) {
		m_build();
	}
	inline FenwickTree(size_t n, const T &value) : m_tree(n + 1, value) {
		m_tree.front() = id_elem;
		m_build();
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline FenwickTree(Iter begin, Iter end)
		: m_tree(std::distance(begin, end) + 1) {
		m_tree.front() = id_elem;
		std::copy(begin, end, m_tree.begin() + 1);
		m_build();
	}

	inline size_t treeSize() const { return m_tree.size(); }
	inline size_t size() const { return (m_tree.size() - 1); }

	inline void resize(size_t n, const T &val = id_elem) {
		if ((++n) <= m_tree.size()) {
			m_tree.resize(n);
		} else {
			size_t old_sz = m_tree.size();
			m_tree.resize(n, val);
			m_rebuild(old_sz);
		}
	}

	inline void assign(std::initializer_list<T> list) {
		assign(list.begin(), list.end());
	}
	inline void assign(size_t n = 0) { m_tree.assign(n + 1, id_elem); }
	inline void assign(size_t n, const T &value) {
		m_tree.assign(n + 1, value);
		m_tree.front() = id_elem;
		m_build();
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline void assign(Iter begin, Iter end) {
		m_tree.resize(std::distance(begin, end) + 1);
		std::copy(begin, end, m_tree.begin() + 1);
		m_build();
	}

	/**
	 * @brief add diff to the element at index
	 */
	inline void modify(size_t index, const T &diff) {
		m_range_check(index);
		for (++index; index < m_tree.size(); index += lowbit(index)) {
			m_tree[index] = m_oper(m_tree[index], diff);
		}
	}

	/**
	 * @return the sum of [0, min(n, size()))
	 */
	inline T query(size_t n = -1) const {
		if (n >= m_tree.size()) {
			n = m_tree.size() - 1;
		}
		T res = id_elem;
		for (; n; n -= lowbit(n)) {
			res = m_oper(res, m_tree[n]);
		}
		return res;
	}
	inline T operator[](size_t n) const { return query(n); }

protected:
	std::vector<T> m_tree; // m_tree[i] maintains the sum of data[i - lowbit(i), i)
	Oper m_oper;

	inline void m_range_check(size_t index) const {
		if (index + 1 >= m_tree.size()) {
			std::__throw_out_of_range_fmt(__N("FenwickTree::__range_check: index "
											  "(which is %zu) >= this->size() "
											  "(which is %zu)"),
										  index, m_tree.size() - 1);
		}
	}
	inline void m_build() {
		for (size_t i = 1, j; i < m_tree.size(); ++i) {
			j = i + lowbit(i);
			if (j < m_tree.size()) {
				m_tree[j] = m_oper(m_tree[j], m_tree[i]);
			}
		}
	}
	inline void m_rebuild(size_t old_tree_sz) {
		for (size_t i = 1, j; i < m_tree.size(); ++i) {
			j = i + lowbit(i);
			if (j >= old_tree_sz && j < m_tree.size()) {
				m_tree[j] = m_oper(m_tree[j], m_tree[i]);
			}
		}
	}
};
```

### Segment Tree

```cpp
size_t n;
std::vector<uint64_t> arr, sum, lzy;

inline size_t leftChild(size_t rt) { return (2 * rt + 1); }
inline size_t rightChild(size_t rt) { return (2 * (rt + 1)); }
inline size_t midOf(size_t beg, size_t end) { return (beg + (end - beg) / 2); }

inline void pushUp(size_t rt) {
	sum[rt] = sum[leftChild(rt)] + sum[rightChild(rt)];
}

inline void pushDown(size_t rt, size_t nd_beg, size_t nd_end) {
	if (!lzy[rt]) {
		return;
	}
	size_t nd_mid = midOf(nd_beg, nd_end);
	size_t lch = leftChild(rt), rch = rightChild(rt);
	sum[lch] += lzy[rt] * (nd_mid - nd_beg);
	sum[rch] += lzy[rt] * (nd_end - nd_mid);
	lzy[lch] += lzy[rt];
	lzy[rch] += lzy[rt];
	lzy[rt] = 0;
}

void build(size_t rt, size_t nd_beg, size_t nd_end) {
	if (nd_beg + 1 == nd_end) {
		sum[rt] = arr[nd_beg];
		return;
	}
	size_t nd_mid = midOf(nd_beg, nd_end);
	build(leftChild(rt), nd_beg, nd_mid);
	build(rightChild(rt), nd_mid, nd_end);
	pushUp(rt);
}

inline void build() {
	sum.resize(4 * n);
	lzy.resize(4 * n);
	build(0, 0, n);
}

void modify(size_t idx, size_t nd_beg, size_t nd_end,
			size_t beg, size_t end, uint64_t diff) {
	if (beg <= nd_beg && nd_end <= end) {
		sum[idx] += (nd_end - nd_beg) * diff;
		lzy[idx] += diff;
		return;
	}
	pushDown(idx, nd_beg, nd_end);
	size_t nd_mid = midOf(nd_beg, nd_end);
	if (beg < nd_mid) {
		modify(leftChild(idx), nd_beg, nd_mid, beg, end, diff);
	}
	if (nd_mid < end) {
		modify(rightChild(idx), nd_mid, nd_end, beg, end, diff);
	}
	pushUp(idx);
}

inline void modify(size_t beg, size_t end, uint64_t diff) {
	modify(0, 0, n, beg, end, diff);
}

uint64_t query(size_t idx, size_t nd_beg, size_t nd_end,
			   size_t beg, size_t end) {
	if (beg <= nd_beg && nd_end <= end) {
		return sum[idx];
	}
	pushDown(idx, nd_beg, nd_end);
	size_t nd_mid = midOf(nd_beg, nd_end);
	uint64_t res = 0;
	if (beg < nd_mid) {
		res += query(leftChild(idx), nd_beg, nd_mid, beg, end);
	}
	if (nd_mid < end) {
		res += query(rightChild(idx), nd_mid, nd_end, beg, end);
	}
	return res;
}

inline uint64_t query(size_t beg, size_t end) {
	return query(0, 0, n, beg, end);
}
```

<div STYLE="page-break-after: always;"></div>

## Number Theorem

### Cantor Expansion

```cpp
/**
 * @return the rank of arr in all its permutation
 */
template <typename T, typename Cmp = std::less<T>, typename Eq = std::equal_to<T>>
uint64_t cantorExpand(const std::vector<T> &arr,
					  Cmp cmp = Cmp(), Eq eq = Eq()) {
	std::vector<size_t> rks;
	size_t n = arr.size(),
		   tot = discretize(arr.begin(), arr.end(), rks, cmp, eq).size();
	uint64_t fact = 1;
	auto bit = FenwickTree<size_t>(tot);
	uint64_t res = 0;
	for (size_t i = 0; i != n; fact *= (++i)) {
		res += fact * bit.query(rks[n - 1 - i]);
		bit.modify(rks[n - 1 - i], 1);
	}
	return res;
}
```

### Quick Power

```cpp
template <typename Uint = uint64_t, typename Aux = __uint128_t,
		  typename = std::enable_if_t<std::is_unsigned<Uint>::value &&
									  std::is_unsigned<Aux>::value &&
									  sizeof(Uint) * 2 <= sizeof(Aux)>>
inline Uint qPowMod(Uint base, Uint exp, Uint mod) {
	Aux mul = base, res = 1;
	while (exp) {
		if (exp & 1) {
			res = res * mul % mod;
		}
		mul = mul * mul % mod;
		exp >>= 1;
	}
	return res;
}
```

### Extended Euclidean Algorithm

```cpp
/**
 * @brief solve the equation a * x + b * y == gcd(a, b)
 * @return gcd(a, b)
 */
inline int64_t exGcd(int64_t a, int64_t b, int64_t &x, int64_t &y) {
	x = 1, y = 0;
	int64_t u = 0, v = 1;
	while (b) {
		int64_t q = a / b;
		std::tie(a, b, x, y, u, v) =
			std::make_tuple(b, a - q * b, u, v, x - q * u, y - q * v);
	}
	return a;
}
```

### Modular Multiplicative Inverse

```cpp
/**
 * @note assuming gcd(x, mod) == 1.
 */
inline int64_t modMulInv(int64_t x, int64_t mod) {
	int64_t res, tmp;
	exGcd(x, mod, res, tmp);
	return ((res % mod + mod) % mod);
}

/**
 * @return modular multiplicative inverse of [0, @c max].
 * @note assuming @c mod is a prime; inverse of 0 is undefined.
 */
std::vector<uint32_t> modMulInvs(uint32_t max, uint64_t mod) {
	std::vector<uint32_t> invs(max + 1);
	invs[1] = 1;
	for (uint32_t i = 2; i <= max; ++i) {
		invs[i] = (mod - mod / i) * invs[mod % i] % mod;
	}
	return invs;
}

/**
 * @return modular multiplicative inverse of each element of @c arr.
 * @note assuming @c mod is a prime; inverse of 0 is undefined.
 */
std::vector<uint64_t> modMulInvs(const std::vector<uint64_t> &arr, uint64_t mod) {
	if (arr.empty()) {
		return arr;
	}
	auto pref_prods = std::vector<uint64_t>(arr.size() + 1);
	pref_prods[0] = 1;
	for (size_t i = 0; i != arr.size(); ++i) {
		pref_prods[i + 1] = pref_prods[i] * arr[i] % mod;
	}
	uint64_t prod_inv = modMulInv(pref_prods.back(), mod);
	auto invs = std::vector<uint64_t>(arr.size());
	for (size_t i = arr.size() - 1; ~i; --i) {
		invs[i] = pref_prods[i] * prod_inv % mod;
		prod_inv = prod_inv * arr[i] % mod;
	}
	return invs;
}
```

### Derangement

```cpp
inline uint64_t derangement(uint64_t n, uint64_t mod) {
	uint64_t d = 1;
	for (uint64_t i = 1; i <= n; ++i) {
		d = (i * d % mod + ((i & 1) ? (mod - 1) : 1)) % mod;
	}
	return d;
}

inline std::vector<uint64_t> derangements(uint64_t max, uint64_t mod) {
	auto d = std::vector<uint64_t>(max + 1);
	d[0] = 1;
	for (uint64_t i = 1; i <= max; ++i) {
		d[i] = (i * d[i - 1] % mod + ((i & 1) ? (mod - 1) : 1)) % mod;
	}
	return d;
}
```

### Prime Sieves

```cpp
std::vector<uint32_t> sieveOfEratosthenes(uint32_t max) {
	if (max <= 1) {
		return {};
	}
	auto not_prime = std::vector<bool>(max + 1);
	std::vector<uint32_t> primes;
	primes.emplace_back(2);
	for (uint32_t i = 4; i <= max; i += 2) {
		not_prime[i] = true;
	}
	for (uint32_t i = 3; i * i <= max; i += 2) {
		if (not_prime[i]) {
			continue;
		}
		for (uint32_t j = (i << 1); j <= max; j += i) {
			not_prime[j] = true;
		}
	}
	for (uint32_t i = 3; i <= max; i += 2) {
		if (!not_prime[i]) {
			primes.emplace_back(i);
		}
	}
	return primes;
}

std::vector<uint32_t> linearSieve(uint32_t max) {
	if (max <= 1) {
		return {};
	}
	auto not_prime = std::vector<bool>(max + 1);
	std::vector<uint32_t> primes;
	primes.emplace_back(2);
	for (uint32_t i = 3; i <= max; i += 2) {
		if (!not_prime[i]) {
			primes.emplace_back(i);
		}
		for (auto j : primes) {
			if (j * i > max) {
				break;
			}
			not_prime[j * i] = true; // j is the minimum prime factor of i * j
			if (i % j == 0) {
				break;
			}
		}
	}
	return primes;
}
```

### Merge Sort Counting Inversions

```cpp
/**
 * @return the number of inversions
 */
template <typename OutputIter, typename AuxIter>
size_t mergeSort(OutputIter first, OutputIter last,
				 AuxIter aux) {
	size_t n = std::distance(first, last);
	if (n <= 1) {
		return 0;
	}
	auto mid = std::next(first, n >> 1);
	size_t inv = mergeSort(first, mid, aux) + mergeSort(mid, last, aux);
	auto i = first, j = mid, k = aux;
	size_t cnt = (n >> 1);
	while (i != mid && j != last) {
		if (*i > *j) {
			*(k++) = *(j++);
			inv += cnt;
		} else {
			*(k++) = *(i++);
			--cnt;
		}
	}
	while (i != mid) {
		*(k++) = *(i++);
	}
	while (j != last) {
		*(k++) = *(j++);
	}
	std::copy(aux, k, first);
	return inv;
}
template <typename OutputIter,
		  typename Container =
			  std::vector<typename std::iterator_traits<OutputIter>::value_type>>
inline size_t mergeSort(OutputIter first, OutputIter last) {
	auto aux = Container(std::distance(first, last));
	return mergeSort(first, last, aux.begin());
}
```

<div STYLE="page-break-after: always;"></div>

## Polynomial

### Fast Fourier Transform

```cpp
template <typename Iter, typename = RequireFwdIter<Iter>>
inline void bitRevPerm(Iter begin, Iter end) {
	size_t n = std::distance(begin, end);
	if (n & (n - 1)) {
		throw std::invalid_argument("bitRevPerm: std::distance(begin, end)"
									" is not a power of 2");
	}
	auto rev = std::vector<size_t>(n);
	for (size_t i = 1; i < n; ++i) {
		rev[i] = (rev[i >> 1] >> 1);
		if (i & 1) {
			rev[i] |= (n >> 1);
		}
	}
	Iter iter = begin;
	for (size_t i = 0; i != n; ++i, ++iter) {
		if (i < rev[i]) {
			std::swap(*iter, *std::next(begin, rev[i]));
		}
	}
}

template <typename Float>
inline void fft(std::vector<std::complex<Float>> &arr, bool inv = false) {
	bitRevPerm(arr.begin(), arr.end());
	for (size_t len = 2; len <= arr.size(); len <<= 1) {
		auto omega = exp(std::complex<Float>(0, (inv ? -2 : 2) * M_PI / len));
		for (size_t i = 0; i != arr.size(); i += len) {
			auto omega_pow = std::complex<Float>(1);
			for (size_t j = 0; (j << 1) != len; ++j) {
				auto even = arr[i + j],
					 odd = omega_pow * arr[i + (len >> 1) + j];
				arr[i + j] = even + odd;
				arr[i + (len >> 1) + j] = even - odd;
				omega_pow *= omega;
			}
		}
	}
	if (inv) {
		for (auto &x : arr) {
			x /= arr.size();
		}
	}
}
```

<div STYLE="page-break-after: always;"></div>

## Sequence

### Discretization

```cpp
/**
 * @return sorted unique elements
 */
template <typename Iter, typename Cmp = std::less<>, typename Eq = std::equal_to<>>
inline std::vector<typename std::iterator_traits<Iter>::value_type>
discretize(Iter begin, Iter end, std::vector<size_t> &res,
		   Cmp cmp = Cmp(), Eq eq = Eq()) {
	auto unq = std::vector<typename std::iterator_traits<Iter>::value_type>(begin, end);
	size_t n = unq.size();
	std::sort(unq.begin(), unq.end(), cmp);
	unq.erase(std::unique(unq.begin(), unq.end(), eq), unq.end());
	res.resize(n);
	for (size_t i = 0; i != n; ++i) {
		res[i] = std::lower_bound(unq.begin(), unq.end(), *(begin++), cmp) - unq.begin();
	}
	return unq;
}
```

### Longest Increasing Subsequence



### Prefix Function

```cpp
template <typename Iter, typename = RequireFwdIter<Iter>>
inline std::vector<size_t> prefFuncOf(Iter begin, Iter end) {
	auto pi = std::vector<size_t>(std::distance(begin, end));
	end = std::next(begin);
	for (size_t i = 1, j; i < pi.size(); ++i, ++end) {
		for (j = pi[i - 1]; j && *end != *std::next(begin, j); j = pi[j - 1])
			;
		pi[i] = j + (*end == *std::next(begin, j));
	}
	return pi;
}

template <typename Iter>
inline std::vector<size_t>
kmp(Iter text_begin, Iter text_end,
	Iter pattern_begin, Iter pattern_end,
	const typename std::iterator_traits<Iter>::value_type &sep = -1) {
	size_t text_sz = std::distance(text_begin, text_end),
		   pattern_sz = std::distance(pattern_begin, pattern_end);
	auto seq = std::vector<typename std::iterator_traits<Iter>::value_type>(
		text_sz + 1 + pattern_sz);
	std::copy(pattern_begin, pattern_end, seq.begin());
	std::copy(text_begin, text_end, seq.end() - text_sz);
	seq[pattern_sz] = sep;
	auto pi = prefFuncOf(seq.begin(), seq.end());
	std::vector<size_t> res;
	for (size_t i = (pattern_sz << 1); i != pi.size(); ++i) {
		if (pi[i] == pattern_sz) {
			res.emplace_back(i - (pattern_sz << 1));
		}
	}
	return res;
}

template <typename Iter>
inline std::vector<size_t> borderLengths(Iter begin, Iter end) {
	auto pi = prefFuncOf(begin, end);
	std::vector<size_t> res;
	for (size_t i = pi.size(); i; i = pi[i - 1]) {
		res.emplace_back(pi[i - 1]);
	}
	return res;
}

/**
 * @note the last period may be incomplete
 */
template <typename Iter>
inline size_t periodLength(Iter begin, Iter end) {
	if (begin == end) {
		return 0;
	}
	auto pi = prefFuncOf(begin, end);
	return (pi.size() - pi.back());
}
```

### Suffix Array

设$\Sigma$为字符集，字符串$\vec s\in\Sigma^n\ (n\in\mathbb N)$。称数组$\vec{sa}\in\mathbb N^n$为字符串$\vec s$的*后缀数组*，当且仅当其满足对任意$i\in\mathbb N\cap[0,n)$，$s[sa_i,n)$是$\vec s$所有后缀中字典序第$i$小的。求后缀数组的同时，记录排名数组$\vec{ra}$，使得对任意$i\in\mathbb N\cap[0,n)$，$ra_i$是$s[i,n)$在$\vec s$所有后缀中按字典序排序后的排名，因此有
$$
sa_{ra_i}=i=ra_{sa_i}
$$

令$lcp(i,j)$表示字符串$\vec s$分别从$i$和$j$开始的后缀的最长公共前缀（longest common prefix）的长度。height数组$\vec h\in\mathbb N^n$定义为
$$
\vec h=(lcp(sa_{i-1},sa_i))_{i=0}^{n-1}
$$
即$h_i$为排名第$i$位的后缀与排在它前面的后缀的LCP长度。特别地，规定$h_0=0$。于是有
$$
h_{ra_i}=lcp(sa_{ra_i},sa_{ra_i-1})=lcp(i,sa_{ra_i-1})
$$
即$h_{ra_i}$是从$i$处开始的后缀与排在它前一位的后缀的LCP长度。

最长公共前缀引理（LCP lemma）：$h_{ra_i}\geq h_{ra_{i-1}}-1$。

```cpp
/**
 * @return {sa, ra, height}
 */
template <typename Iter>
inline std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
sufArrOf(Iter begin, Iter end) {
	std::vector<size_t> ra;
	size_t unq = discretize(begin, end, ra).size(), n = ra.size();
	auto cnt = std::vector<size_t>(n);
	auto sa = std::vector<size_t>(n),
		 height = std::vector<size_t>(n),
		 tmp = std::vector<size_t>(n);

	for (size_t i = 0; i < n; ++i) {
		++cnt[ra[i]];
	}
	for (size_t i = 1; i < unq; ++i) {
		cnt[i] += cnt[i - 1];
	}
	for (size_t i = n; (i--) > 0;) {
		sa[--cnt[ra[i]]] = i;
	}

	for (size_t len = 1, tot; len < n; len <<= 1) {
		// Sort sa so that ra[sa[i] + len] <= ra[sa[i + 1] + len]
		tot = 0;
		for (size_t i = n - len; i != n; ++i) {
			tmp[tot++] = i;
		}
		for (size_t i = 0; i != n; ++i) {
			if (sa[i] >= len) {
				tmp[tot++] = sa[i] - len;
			}
		}

		// Stably sort sa so that ra[sa[i]] <= ra[sa[i + 1]]
		std::fill(cnt.begin(), cnt.begin() + unq, 0);
		for (auto i : tmp) {
			++cnt[ra[i]];
		}
		for (size_t i = 1; i != unq; ++i) {
			cnt[i] += cnt[i - 1];
		}
		for (size_t i = n; (i--) > 0;) {
			sa[--cnt[ra[tmp[i]]]] = tmp[i];
		}

		// Update ra
		std::copy(ra.begin(), ra.end(), tmp.begin());
		ra[sa[0]] = 0;
		tot = 1;
		for (size_t i = 1; i != n; ++i) {
			if (tmp[sa[i]] == tmp[sa[i - 1]] &&
				(sa[i] + len < n) == (sa[i - 1] + len < n) &&
				((sa[i] + len < n)
					 ? (tmp[sa[i] + len] == tmp[sa[i - 1] + len])
					 : true)) {
				ra[sa[i]] = tot - 1;
			} else {
				ra[sa[i]] = (tot++);
			}
		}
		if ((unq = tot) == n) {
			break;
		}
	}

	// Calculate height
	for (size_t i = 0, j, len = 0; i != n; ++i) {
		if (!ra[i]) {
			len = 0;
			continue;
		}
		if (len) {
			--len;
		}
		for (j = sa[ra[i] - 1];
			 *std::next(begin, i + len) == *std::next(begin, j + len);
			 ++len)
			;
		height[ra[i]] = len;
	}

	return {sa, ra, height};
}
```

### Hash

```cpp
template <typename Mod = uint32_t,
		  typename Aux = uint64_t,
		  typename = std::enable_if_t<std::is_unsigned<Mod>::value &&
									  std::is_unsigned<Aux>::value &&
									  sizeof(Mod) * 2 == sizeof(Aux)>>
std::vector<Mod> hashOf(const std::string &s,
						Mod base = 233, Mod mod = 993244853) {
	auto h = std::vector<Mod>(s.size() + 1);
	Aux b = base;
	for (size_t i = 0; i != s.size(); ++i) {
		h[i + 1] = (h[i] * b + s[i]) % mod;
	}
	return h;
}
```

### Manacher

```cpp
/**
 * @return {odd, even}, both denote the number of palindrome subsequences
 * 	centering around each elements (consider the right one as the center for
 * 	even-length palindromes)
 */
template <typename Iter, typename = RequireFwdIter<Iter>>
inline std::pair<std::vector<size_t>, std::vector<size_t>>
manacher(Iter begin, Iter end) {
	size_t n = std::distance(begin, end);
	if (!n) {
		return {{}, {}};
	}
	auto odd = std::vector<size_t>(n), even = std::vector<size_t>(n);
	odd[0] = 1;
	for (size_t i = 1, b = 0, e = 1; i != n; ++i) {
		if (i < e) {
			odd[i] = std::min(odd[b + e - 1 - i], e - i);
		}
		while (odd[i] <= i && i + odd[i] < n &&
			   *std::next(begin, i - odd[i]) ==
				   *std::next(begin, i + odd[i])) {
			++odd[i];
		}
		if (i + odd[i] > e) {
			b = i - odd[i] + 1;
			e = i + odd[i];
		}
	}
	for (size_t i = 0, b = 0, e = 0; i != n; ++i) {
		if (i < e) {
			even[i] = std::min(even[b + e - i], e - i);
		}
		while (even[i] < i && i + even[i] < n &&
			   *std::next(begin, i - even[i] - 1) ==
				   *std::next(begin, i + even[i])) {
			++even[i];
		}
		if (i + even[i] > e) {
			b = i - even[i];
			e = i + even[i];
		}
	}
	return {odd, even};
}
```

<div STYLE="page-break-after: always;"></div>

## Graph Theory

```cpp
template <typename W>
struct EdgeImpl {
	size_t u, v;
	W w;
};
template <>
struct EdgeImpl<void> {
	size_t u, v;
};

template <typename Weight = int64_t, bool is_directed = true>
class Graph {
public:
	// When Weight is void, Edge does not has member w
	using Edge = EdgeImpl<Weight>;

	Graph(size_t n = 0) : m_adj(n) {}
	Graph(size_t n, const std::vector<Edge> &edges)
		: m_adj(n), m_edges(edges) { m_insertEdgesToAdj(); }

	void assign(size_t n) {
		m_edges.clear();
		m_adj.resize(n);
		for (auto &lst : m_adj) {
			lst.clear();
		}
	}
	void assign(size_t n, const std::vector<Edge> &edges) {
		m_edges = edges;
		m_adj.resize(n);
		for (auto &lst : m_adj) {
			lst.clear();
		}
		m_insertEdgesToAdj();
	}

	template <typename W = Weight>
	typename std::enable_if<std::is_void<W>::value, void>::type
	insertEdge(size_t u, size_t v) {
		m_edges.push_back(Edge{u, v});
		m_insertToAdj(m_edges.size() - 1);
	}
	template <typename W = Weight,
			  typename = typename std::enable_if<std::is_same<W, Weight>::value>::type>
	typename std::enable_if<!std::is_void<W>::value, void>::type
	insertEdge(size_t u, size_t v, const W &w) {
		m_edges.push_back(Edge{u, v, w});
		m_insertToAdj(m_edges.size() - 1);
	}

	void reserve(size_t n, size_t m) {
		m_edges.reserve(m);
		m_adj.reserve(n);
	}

	void clear() {
		m_edges.clear();
		m_adj.clear();
	}

	std::pair<size_t, size_t> size() const {
		return std::make_pair(m_adj.size(), m_edges.size());
	}

	const std::vector<Edge> &edges() const { return m_edges; }
	const std::vector<std::vector<size_t>> &adj() const { return m_adj; }

	/* Algorithms */
	std::vector<std::vector<size_t>> tarjanSccs() const;
	/**
	 * @return {{cut_verts, vbccs}, {bridges, ebccs}}
	 */
	inline std::pair<std::pair<std::vector<std::vector<size_t>>,
							   std::vector<std::vector<size_t>>>,
					 std::pair<std::vector<std::vector<size_t>>,
							   std::vector<std::vector<size_t>>>>
	tarjanCutAndBccs() const;

	std::vector<size_t> toposort() const;

	std::vector<std::vector<size_t>> kruskal() const;
	std::vector<size_t> prim(size_t rt = 0) const;

	std::vector<Weight> dijkstra(size_t src) const;
	std::vector<Weight> bellmanFord(size_t src) const;
	std::vector<Weight> spfa(size_t src) const;

protected:
	std::vector<Edge> m_edges;
	std::vector<std::vector<size_t>> m_adj;

	void m_insertToAdj(size_t idx) {
		m_adj[m_edges[idx].u].emplace_back(idx);
		if (m_edges[idx].u != m_edges[idx].v) {
			m_adj[m_edges[idx].v].emplace_back(idx);
		}
	}
	void m_insertEdgesToAdj() {
		for (size_t i = 0; i != m_edges.size(); ++i) {
			m_insertToAdj(i);
		}
	}

private:
};
```

### Tarjan

#### Tarjan for Strong Connected Components
```cpp
template <typename Weight, bool is_directed>
std::vector<std::vector<size_t>>
Graph<Weight, is_directed>::tarjanSccs() const {
	static_assert(is_directed,
				  "Tarjan's algorithm for strongly connected components "
				  "is only applicable to directed graphs.");
	std::vector<size_t> stk;
	stk.reserve(m_adj.size());
	std::vector<bool> in_stk(m_adj.size());
	std::vector<size_t> dfn(m_adj.size(), size_t(-1)), low(m_adj.size());
	std::vector<std::vector<size_t>> sccs;
	size_t tm = 0;
	std::function<void(size_t)> dfs = [&](size_t cur) -> void {
		stk.emplace_back(cur);
		in_stk[cur] = true;
		low[cur] = (dfn[cur] = (tm++));
		for (auto i : m_adj[cur]) {
			auto &edge = m_edges[i];
			if (edge.u != cur) { // edges pointing to cur
				continue;
			}
			if (dfn[edge.v] == size_t(-1)) {
				dfs(edge.v);
				minEq(low[cur], low[edge.v]);
			} else if (in_stk[edge.v]) {
				minEq(low[cur], dfn[edge.v]);
			}
		}
		if (dfn[cur] == low[cur]) {
			sccs.emplace_back();
			size_t vert;
			do {
				sccs.back().emplace_back(vert = stk.back());
				stk.pop_back();
				in_stk[vert] = false;
			} while (vert != cur);
		}
	};
	for (size_t i = 0; i != m_adj.size(); ++i) {
		if (dfn[i] == size_t(-1)) {
			dfs(i);
		}
	}
	return sccs;
}
```

#### Tarjan for Cuts and Biconnected Components

```cpp
template <typename Weight, bool is_directed>
inline std::pair<std::pair<std::vector<std::vector<size_t>>,
						   std::vector<std::vector<size_t>>>,
				 std::pair<std::vector<std::vector<size_t>>,
						   std::vector<std::vector<size_t>>>>
Graph<Weight, is_directed>::tarjanCutAndBccs() const {
	static_assert(!is_directed,
				  "Tarjan's algorithm for cut vertices, bridges "
				  "and biconnected components is only applicable to "
				  "undirected graphs.");
	std::vector<std::vector<size_t>> cut_verts, vbccs, bridges, ebccs;
	std::vector<size_t> vbcc_stk, ebcc_stk;
	std::vector<size_t> dfn(m_adj.size(), size_t(-1)), low(m_adj.size());
	size_t tm = 0;
	std::function<void(size_t, size_t)> dfs = [&](size_t pre, size_t cur) {
		low[cur] = (dfn[cur] = (tm++));
		vbcc_stk.emplace_back(cur);
		ebcc_stk.emplace_back(cur);
		size_t children = 0;
		bool not_in_cut = true, has_edge_to_pre = false;
		for (auto i : m_adj[cur]) {
			size_t nxt = ((m_edges[i].u == cur) ? m_edges[i].v : m_edges[i].u);
			if (dfn[nxt] == size_t(-1)) {
				++children;
				dfs(cur, nxt);
				minEq(low[cur], low[nxt]);
				if (~pre ? (low[nxt] >= dfn[cur]) : (children > 1)) {
					if (not_in_cut) {
						not_in_cut = false;
						cut_verts.back().emplace_back(cur);
					}
					vbccs.emplace_back(1, cur);
					size_t vert;
					do {
						vbccs.back().push_back(vert = vbcc_stk.back());
						vbcc_stk.pop_back();
					} while (vert != nxt);
				}
				if (low[nxt] > dfn[cur]) {
					bridges.back().emplace_back(i);
					ebccs.emplace_back();
					size_t vert;
					do {
						ebccs.back().emplace_back(vert = ebcc_stk.back());
						ebcc_stk.pop_back();
					} while (vert != nxt);
				}
			} else if (nxt != pre || has_edge_to_pre) {
				minEq(low[cur], dfn[nxt]);
			} else {
				has_edge_to_pre = true;
			}
		}
	};
	for (size_t i = 0; i != m_adj.size(); ++i) {
		if (dfn[i] == size_t(-1)) {
			cut_verts.emplace_back();
			bridges.emplace_back();
			dfs(-1, i);
			if (vbcc_stk.size()) {
				vbccs.emplace_back(std::move(vbcc_stk));
			}
			if (ebcc_stk.size()) {
				ebccs.emplace_back(std::move(ebcc_stk));
			}
		}
	}
	return {{cut_verts, vbccs}, {bridges, ebccs}};
}
```

### Topological Sort

```cpp
template <typename Weight, bool is_directed>
inline std::vector<size_t> Graph<Weight, is_directed>::toposort() const {
	static_assert(is_directed,
				  "Topological sorting is only applicable to directed graphs.");
	std::vector<size_t> indeg(m_adj.size());
	for (auto &edge : m_edges) {
		++indeg[edge.v];
	}
	std::vector<size_t> res, zero_indeg_verts;
	for (size_t i = 0; i != m_adj.size(); ++i) {
		if (!indeg[i]) {
			zero_indeg_verts.emplace_back(i);
		}
	}
	while (zero_indeg_verts.size()) {
		size_t frm = zero_indeg_verts.back();
		zero_indeg_verts.pop_back();
		res.emplace_back(frm);
		for (auto i : m_adj[frm]) {
			if (!(--indeg[m_edges[i].v])) {
				zero_indeg_verts.emplace_back(m_edges[i].v);
			}
		}
	}
	return ((res.size() == m_adj.size()) ? res : std::vector<size_t>());
}
```

### Minimum Spanning Tree and Forest

#### Kruskal

```cpp
template <typename Weight, bool is_directed>
inline std::vector<std::vector<size_t>>
Graph<Weight, is_directed>::kruskal() const {
	static_assert(!is_directed,
				  "Kruskal's algorithm is only applicable to undirected graphs.");
	if (m_adj.size() < 2) {
		return {};
	}

	std::vector<size_t> sorted(m_edges.size());
	std::iota(sorted.begin(), sorted.end(), 0);
	std::sort(sorted.begin(), sorted.end(),
			  [&](size_t lhs, size_t rhs) {
				  return (m_edges[lhs].w < m_edges[rhs].w);
			  });

	std::vector<size_t> dsu(m_adj.size());
	std::iota(dsu.begin(), dsu.end(), 0);
	auto find = [&](size_t x) -> size_t {
		while (dsu[dsu[x]] != dsu[x]) {
			dsu[x] = dsu[dsu[x]];
		}
		return dsu[x];
	};
	auto merge = [&](size_t to, size_t frm) { dsu[find(frm)] = find(to); };

	std::vector<size_t> msf_edges;
	msf_edges.reserve(m_adj.size() - 1);
	for (auto i : sorted) {
		if (find(m_edges[i].u) != find(m_edges[i].v)) {
			msf_edges.emplace_back(i);
			merge(m_edges[i].u, m_edges[i].v);
		}
	}

	// Classify edges into connected components
	std::unordered_map<size_t, std::vector<size_t>> components;
	for (auto i : msf_edges) {
		components[find(m_edges[i].u)].emplace_back(i);
	}
	std::vector<std::vector<size_t>> res;
	res.reserve(components.size());
	for (auto &pr : components) {
		res.emplace_back(std::move(pr.second));
	}
	return res;
}
```

#### Prim

```cpp
template <typename Weight, bool is_directed>
inline std::vector<size_t> Graph<Weight, is_directed>::prim(size_t rt) const {
	static_assert(!is_directed,
				  "Prim algorithm is only applicable to undirected graphs.");
	if (m_adj.size() < 2) {
		return {};
	}

	std::vector<size_t> mst_edges;
	mst_edges.reserve(m_adj.size() - 1);
	auto cmp = [&](size_t lhs, size_t rhs) {
		return (m_edges[lhs].w > m_edges[rhs].w);
	};
	std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> pq(cmp);
	std::vector<bool> vis(m_adj.size());

	auto visit = [&](size_t frm) -> void {
		vis[frm] = true;
		for (auto i : m_adj[frm]) {
			const auto &edge = m_edges[i];
			if (!vis[(edge.u == frm ? edge.v : edge.u)]) {
				pq.push(i);
			}
		}
	};

	visit(rt);
	while (pq.size() && mst_edges.size() + 1 < m_adj.size()) {
		size_t i = pq.top();
		pq.pop();
		size_t u = m_edges[i].u, v = m_edges[i].v;
		if (vis[u] && vis[v]) {
			continue;
		}
		mst_edges.emplace_back(i);
		visit(vis[u] ? v : u);
	}

	return mst_edges;
}
```

### Shortest Path

#### Floyd

```cpp
template <typename Weight>
inline std::vector<std::vector<Weight>>
floyd(std::vector<std::vector<Weight>> weights) {
	size_t n = weights.size();
	if (!n) {
		return {};
	}
	assert(n == weights[0].size());
	for (size_t i = 0, j, k; i != n; ++i) {
		for (j = 0; j != n; ++j) {
			for (k = 0; k != n; ++k) {
				if (weights[j][i] != std::numeric_limits<Weight>::max() &&
					weights[i][k] != std::numeric_limits<Weight>::max()) {
					weights[j][k] = std::min(weights[j][k], weights[j][i] + weights[i][k]);
				}
			}
		}
	}
	return weights;
}

template <typename Weight>
inline std::vector<std::vector<Weight>>
floydMatMul(std::vector<std::vector<Weight>> weights) {
	size_t n = weights.size();
	if (!n) {
		return {};
	}
	assert(n == weights[0].size());
	using Mat = std::vector<std::vector<Weight>>;
	auto matMul = [](const Mat &lhs, const Mat &rhs) -> Mat {
		auto &&n = lhs.size();
		auto res = Mat(n, std::vector<Weight>(n, std::numeric_limits<Weight>::max()));
		for (size_t i = 0, j, k; i != n; ++i) {
			for (j = 0; j != n; ++j) {
				for (k = 0; k != n; ++k) {
					if (lhs[i][k] != std::numeric_limits<Weight>::max() &&
						rhs[k][j] != std::numeric_limits<Weight>::max()) {
						res[i][j] = std::min(res[i][j], lhs[i][k] + rhs[k][j]);
					}
				}
			}
		}
		return res;
	};
	auto res = Mat(n, std::vector<int32_t>(n, std::numeric_limits<Weight>::max()));
	for (size_t i = 0; i != n; ++i) {
		res[i][i] = 0;
	}
	for (; n; n >>= 1) {
		if (n & 1) {
			res = matMul(res, weights);
		}
		weights = matMul(weights, weights);
	}
	return res;
}
```

#### Dijkstra

```cpp
template <typename Weight, bool is_directed>
inline std::vector<Weight> Graph<Weight, is_directed>::dijkstra(size_t src) const {
	using Adj = std::pair<Weight, size_t>; // (dist, vertex)
	std::priority_queue<Adj, std::vector<Adj>, std::greater<Adj>> pq;
	std::vector<Weight> dist(m_adj.size(), std::numeric_limits<Weight>::max());
	std::vector<bool> vis(m_adj.size());

	pq.emplace(0, src);
	dist[src] = 0;
	while (pq.size()) {
		auto frm = pq.top().second;
		pq.pop();
		if (vis[frm]) {
			continue;
		}
		vis[frm] = true;
		for (auto i : m_adj[frm]) {
			auto to = (is_directed
						   ? m_edges[i].v
						   : ((m_edges[i].u == frm) ? m_edges[i].v : m_edges[i].u));
			auto w = m_edges[i].w;
			if (dist[to] > dist[frm] + w) {
				pq.emplace(dist[to] = dist[frm] + w, to);
			}
		}
	}

	return dist;
}
```

#### Bellman-Ford

```cpp
template <typename Weight, bool is_directed>
inline std::vector<Weight>
Graph<Weight, is_directed>::bellmanFord(size_t src) const {
	std::vector<Weight> dist(m_adj.size(), std::numeric_limits<Weight>::max());
	auto relax = [&](size_t u, size_t v, Weight w) -> bool {
		if (dist[u] != std::numeric_limits<Weight>::max() &&
			dist[v] > dist[u] + w) {
			dist[v] = dist[u] + w;
			return false;
		}
		return true;
	};

	dist[src] = 0;
	for (size_t cnt = 0; cnt != m_adj.size(); ++cnt) {
		bool flag = true;
		for (auto &edge : m_edges) {
			flag &= relax(edge.u, edge.v, edge.w);
			if (!is_directed) {
				flag &= relax(edge.v, edge.u, edge.w);
			}
		}
		if (flag) {
			return dist;
		}
	}
	return {}; // negative cycle detected
}
```

#### SPFA

```cpp
template <typename Weight, bool is_directed>
inline std::vector<Weight>
Graph<Weight, is_directed>::spfa(size_t src) const {
	std::vector<Weight> dist(m_adj.size(), std::numeric_limits<Weight>::max());
	std::vector<bool> inq(m_adj.size());
	std::vector<size_t> cnt(m_adj.size());
	std::queue<size_t> q;

	dist[src] = 0;
	q.emplace(src);
	while (q.size()) {
		auto frm = q.front();
		inq[frm] = false;
		q.pop();
		for (auto i : m_adj[frm]) {
			auto to = (is_directed
						   ? m_edges[i].v
						   : ((m_edges[i].u == frm) ? m_edges[i].v : m_edges[i].u));
			auto w = m_edges[i].w;
			if (dist[to] > dist[frm] + w) {
				dist[to] = dist[frm] + w;
				/**
				 * the shortest path between 2 vertices consists of at most
				 * (n - 1) edgess
				 */
				if ((cnt[to] = cnt[frm] + 1) >= m_adj.size()) {
					return {};
				}
				if (!inq[to]) {
					inq[to] = true;
					q.emplace(to);
				}
			}
		}
	}
	return dist;
}
```

### Shortest Hamiltonian Path and Cycle

```cpp
/**
 * @return the shortest Hamiltonian cycle beginning and ending at vertex 0
 */
template <typename Weight>
inline std::vector<size_t>
shortestHamiltonianCycle(const std::vector<std::vector<Weight>> &adj_mat,
						 Weight inf = std::numeric_limits<Weight>::max()) {
	size_t n = adj_mat.size();
	if (!n) {
		return {};
	}
	if (n == 1) {
		return {0};
	}
	auto dp = std::vector<std::vector<Weight>>(uint64_t(1) << n,
											   std::vector<Weight>(n, inf));
	auto pre = std::vector<std::vector<size_t>>(
		uint64_t(1) << n,
		std::vector<size_t>(n, size_t(-1)));
	dp[1][0] = 0;
	for (uint64_t set = 1; set < (uint64_t(1) << n); set += 2) {
		for (size_t cur = (set != 1); cur < n; ++cur) {
			if (!(set & (1 << cur))) {
				continue;
			}
			for (size_t nxt = 1; nxt < n; ++nxt) {
				if ((set & (1 << nxt)) || adj_mat[cur][nxt] == inf) {
					continue;
				}
				uint64_t nxt_set = set | (1 << nxt);
				if (dp[set][cur] + adj_mat[cur][nxt] < dp[nxt_set][nxt]) {
					pre[nxt_set][nxt] = cur;
					dp[nxt_set][nxt] = dp[set][cur] + adj_mat[cur][nxt];
				}
			}
		}
	}
	uint64_t full_set = (uint64_t(1) << n) - 1;
	size_t last = 1;
	for (size_t i = 2; i < n; ++i) {
		if (adj_mat[i][0] != inf &&
			dp[full_set][i] + adj_mat[i][0] <
				dp[full_set][last] + adj_mat[last][0]) {
			last = i;
		}
	}
	std::vector<size_t> res;
	res.reserve(n);
	while (full_set) {
		res.emplace_back(last);
		size_t pre_last = pre[full_set][last];
		full_set ^= (1 << last);
		last = pre_last;
	}
	std::reverse(res.begin(), res.end());
	return res;
}

template <typename Weight>
inline std::vector<size_t>
shortestHamiltonianPath(const std::vector<std::vector<Weight>> &adj_mat,
						size_t src = 0,
						Weight inf = std::numeric_limits<Weight>::max()) {

	size_t n = adj_mat.size();
	if (!n) {
		return {};
	}
	if (n == 1) {
		return {0};
	}
	auto dp = std::vector<std::vector<Weight>>(uint64_t(1) << n,
											   std::vector<Weight>(n, inf));
	auto pre = std::vector<std::vector<size_t>>(uint64_t(1) << n,
												std::vector<size_t>(n, size_t(-1)));
	if (src != size_t(-1)) {
		dp[uint64_t(1) << src][src] = 0;
	} else {
		for (size_t i = 0; i < n; ++i) {
			dp[uint64_t(1) << i][i] = 0;
		}
	}
	for (uint64_t set = 1; set < (uint64_t(1) << n); ++set) {
		for (size_t cur = 0; cur < n; ++cur) {
			if (!(set & (uint64_t(1) << cur))) {
				continue;
			}
			if (dp[set][cur] == inf) {
				continue;
			}
			for (size_t nxt = 0; nxt < n; ++nxt) {
				if ((set & (uint64_t(1) << nxt)) || adj_mat[cur][nxt] == inf) {
					continue;
				}
				uint64_t nxt_set = set | (uint64_t(1) << nxt);
				if (dp[set][cur] + adj_mat[cur][nxt] < dp[nxt_set][nxt]) {
					dp[nxt_set][nxt] = dp[set][cur] + adj_mat[cur][nxt];
					pre[nxt_set][nxt] = cur;
				}
			}
		}
	}
	uint64_t full_set = (uint64_t(1) << n) - 1;
	size_t last = size_t(-1);
	Weight min_path_len = inf;

	for (size_t i = 0; i < n; ++i) {
		if (i == last || dp[full_set][i] < min_path_len) {
			min_path_len = dp[full_set][i];
			last = i;
		}
	}
	if (last == size_t(-1)) {
		return {};
	}
	std::vector<size_t> res;
	res.reserve(n);
	uint64_t current_set = full_set;
	while (last != size_t(-1)) {
		res.emplace_back(last);
		size_t pre_last = pre[full_set][last];
		full_set ^= (uint64_t(1) << last);
		last = pre_last;
	}
	std::reverse(res.begin(), res.end());
	return res;
}
```