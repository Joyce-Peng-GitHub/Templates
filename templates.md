# Templates

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

## Data Structures

### SparseTable

```cpp
template <typename T, typename Cmp = std::less<>>
struct Min {
	inline const T &operator()(const T &lhs, const T &rhs) const {
		return std::min(lhs, rhs, Cmp());
	}
}

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
}
```

### MonoDeque

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
}
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
}
```

### Segment Tree

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

## String

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
	std::vector<std::vector<size_t>> tarjanSccs() const {
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
	/**
	 * @return {{cut_verts, vbccs}, {bridges, ebccs}}
	 */
	inline std::pair<std::pair<std::vector<std::vector<size_t>>,
							   std::vector<std::vector<size_t>>>,
					 std::pair<std::vector<std::vector<size_t>>,
							   std::vector<std::vector<size_t>>>>
	tarjanCutAndBccs() const {
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
				size_t nxt = ((m_edges[i].u == cur)
								  ? m_edges[i].v
								  : m_edges[i].u);
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
}
```