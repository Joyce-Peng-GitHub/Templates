#include <bits/stdc++.h>

template <typename Int,
		  typename = typename std::enable_if<std::is_integral<Int>::value>::type>
inline void readInt(Int &res, FILE *file = stdin) {
	res = 0;
	bool neg = false;
	int ch = std::fgetc(file);
	while (~ch && std::isspace(ch)) ch = std::fgetc(file);
	if (ch == '-') {
		neg = true;
		ch = std::fgetc(file);
	} else if (ch == '+') {
		ch = std::fgetc(file);
	}
	while (std::isdigit(ch)) {
		res = (res << 1) + (res << 3) + (ch ^ '0');
		ch = std::fgetc(file);
	}
	if (neg) res = -res;
}

template <typename Uint,
		  typename = typename std::enable_if<std::is_integral<Uint>::value &&
											 std::is_unsigned<Uint>::value>::type>
inline void writeUint(Uint val, FILE *file = stdout) {
	if (!val) {
		std::fputc('0', file);
		return;
	}
	std::array<char, 39> buf;
	size_t cnt = 0;
	while (val) {
		buf[cnt++] = ((val % 10) ^ '0');
		val /= 10;
	}
	while (cnt--) std::fputc(buf[cnt], file);
}
template <typename Int,
		  typename = typename std::enable_if<std::is_integral<Int>::value>::type>
inline void writeInt(Int val, FILE *file = stdout) {
	if (val < 0) {
		std::fputc('-', file);
		writeUint(-static_cast<typename std::make_unsigned<Int>::type>(val), file);
	} else {
		writeUint(static_cast<typename std::make_unsigned<Int>::type>(val), file);
	}
}

inline __uint128_t stoulll(const std::string &str, std::size_t *pos = nullptr, int base = 10) {
	auto iter = str.begin();
	while (iter != str.end() && std::isspace(*iter)) {
		++iter;
	}
	bool neg = false;
	if (*iter == '-') {
		neg = true;
		++iter;
	} else if (*iter == '+') {
		++iter;
	}
	if (base == 0) {
		if (*iter == '0') {
			if (iter[1] == 'x' || iter[1] == 'X') {
				base = 16;
				iter += 2;
			} else {
				base = 8;
			}
		} else {
			base = 10;
		}
	} else if (base == 16 &&
			   *iter == '0' &&
			   iter + 2 < str.end() &&
			   (iter[1] == 'x' || iter[1] == 'X')) {
		iter += 2;
	}
	if (base < 2 || base > 26) {
		throw std::invalid_argument("stoulll");
	}
	__uint128_t res = 0, max_val = static_cast<__uint128_t>(-1);
	bool has_digit = false;
	while (*iter) {
		int digit;
		if (std::isdigit(*iter)) {
			digit = (*iter ^ '0');
		} else if (std::isalpha(*iter)) {
			digit = std::toupper(*iter) - 'A' + 10;
		} else {
			break;
		}
		if (digit >= base) break;
		if (res > (max_val - digit) / base) throw std::out_of_range("stoulll");
		res = res * base + digit;
		has_digit = true;
		++iter;
	}
	if (!has_digit) throw std::invalid_argument("stoulll");
	if (pos) *pos = iter - str.begin();
	return (neg ? (-res) : res);
}

inline __int128_t stolll(const std::string &str, std::size_t *pos = nullptr, int base = 10) {
	auto iter = str.begin();
	while (iter != str.end() && std::isspace(*iter)) ++iter;
	bool neg = false;
	if (*iter == '-') {
		neg = true;
		++iter;
	} else if (*iter == '+') {
		++iter;
	}
	if (base == 0) {
		if (*iter == '0') {
			if (iter[1] == 'x' || iter[1] == 'X') {
				base = 16;
				iter += 2;
			} else {
				base = 8;
			}
		} else {
			base = 10;
		}
	} else if (base == 16 &&
			   *iter == '0' &&
			   iter + 2 < str.end() &&
			   (iter[1] == 'x' || iter[1] == 'X')) {
		iter += 2;
	}
	if (base < 2 || base > 26) {
		throw std::invalid_argument("stoulll");
	}
	__int128_t res = 0,
			   min_val = (static_cast<__uint128_t>(1) << 127),
			   max_val = (static_cast<__uint128_t>(-1) >> 1);
	bool has_digit = false;
	while (*iter) {
		int digit;
		if (std::isdigit(*iter)) {
			digit = (*iter ^ '0');
		} else if (std::isalpha(*iter)) {
			digit = std::toupper(*iter) - 'A' + 10;
		} else {
			break;
		}
		if (digit >= base) break;
		if (neg) {
			if (res < (min_val + digit) / base) {
				throw std::out_of_range("stoulll");
			}
			res = res * base - digit;
		} else {
			if (res > (max_val - digit) / base) {
				throw std::out_of_range("stoulll");
			} else {
				res = res * base + digit;
			}
		}
		has_digit = true;
		++iter;
	}
	if (!has_digit) throw std::invalid_argument("stoulll");
	if (pos) *pos = iter - str.begin();
	return res;
}

template <typename T, typename Cmp = std::less<T>>
inline T &minEq(T &lhs, const T &rhs, Cmp cmp = Cmp()) {
	return (cmp(rhs, lhs) ? (lhs = rhs) : lhs);
}

template <typename T, typename Cmp = std::less<T>>
inline T &maxEq(T &lhs, const T &rhs, Cmp cmp = Cmp()) {
	return (cmp(lhs, rhs) ? (lhs = rhs) : lhs);
}

/**
 * @brief behaves as if it generates func[beg, end) and
 * 	performs std::lower_bound on it
 */
template <typename T, typename Ret, typename Func,
		  typename Cmp = std::less<Ret>>
inline T lowerBound(T beg, T end, const Ret &val,
					Func func = Func(), Cmp cmp = Cmp()) {
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
inline T upperBound(T beg, T end, const Ret &val,
					Func func = Func(), Cmp cmp = Cmp()) {
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

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
	os << '[';
	if (vec.size()) {
		os << vec.front();
		for (size_t i = 1; i != vec.size(); ++i) {
			os << ',' << ' ' << vec[i];
		}
	}
	return (os << ']');
}

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

namespace std {
	template <>
	struct is_unsigned<__uint128_t> : public true_type {};
}

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

class Dsu {
public:
	Dsu(size_t n = 0) : m_fa_or_sz(n, -1) {}

	void assign(size_t n) { m_fa_or_sz.assign(n, -1); }
	void clear() { m_fa_or_sz.clear(); }
	void resize(size_t n) { m_fa_or_sz.resize(n, -1); }

	size_t size() const noexcept { return m_fa_or_sz.size(); }
	bool empty() const noexcept { return m_fa_or_sz.empty(); }

	size_t leader(size_t x) {
		if (m_fa_or_sz[x] < 0) return x;
		return (m_fa_or_sz[x] = leader(m_fa_or_sz[x]));
	}
	bool same(size_t x, size_t y) { return (leader(x) == leader(y)); }

	size_t size(size_t x) { return -m_fa_or_sz[leader(x)]; }

	size_t merge(size_t x, size_t y) {
		x = leader(x);
		y = leader(y);
		if (x == y) return x;
		if (-m_fa_or_sz[x] < -m_fa_or_sz[y]) std::swap(x, y);
		m_fa_or_sz[x] += m_fa_or_sz[y];
		m_fa_or_sz[y] = x;
		return x;
	}

	std::vector<std::vector<size_t>> groups() {
		std::vector<size_t> leader_of(size()), sz(size());
		for (size_t i = 0; i != size(); ++i) ++sz[leader_of[i] = leader(i)];
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
		if (len == 0) throw std::out_of_range("SparseTable::query: len == 0");
		if (pos + len > size()) len = size() - pos;
		size_t log_len = floorLogn2(len);
		return m_oper(m_data[pos][log_len],
					  m_data[pos + len - (size_t(1) << log_len)][log_len]);
	}

protected:
	Oper m_oper;
	Table m_data; // m_data[i][j] maintains Oper(data[i, i + 2 ** j))

private:
};

template <typename T>
class Deque {
public:
	using value_type = T;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;
	using reference = T &;
	using const_reference = const T &;
	using pointer = T *;
	using const_pointer = const T *;

protected:
	template <bool IsConst>
	class IterBase {
	public:
		// C++ standard iterator traits
		using iterator_category = std::random_access_iterator_tag;
		using value_type = T;
		using difference_type = std::ptrdiff_t;

		// Use std::conditional to select const or non-const types
		using DequeType = typename std::conditional<IsConst, const Deque, Deque>::type;
		using pointer = typename std::conditional<IsConst, const T *, T *>::type;
		using reference = typename std::conditional<IsConst, const T &, T &>::type;

	protected:
		DequeType *m_dq = nullptr;
		size_type m_idx = 0; // Logical idx from the start of the deque

		// Friend declaration to allow Deque to access private members.
		friend class Deque<T>;

	public:
		IterBase() = default;
		IterBase(DequeType *dq, size_type idx) : m_dq(dq), m_idx(idx) {}

		// Allows implicit conversion from iterator to const_iterator
		operator IterBase<true>() const { return IterBase<true>(m_dq, m_idx); }

		// Dereference operators
		reference operator*() const { return (*m_dq)[m_idx]; }
		pointer operator->() const { return &((*m_dq)[m_idx]); }
		reference operator[](difference_type offset) const {
			return (*m_dq)[m_idx + offset];
		}

		// Increment and decrement operators
		IterBase &operator++() {
			++m_idx;
			return *this;
		}
		IterBase operator++(int) {
			IterBase tmp = *this;
			++m_idx;
			return tmp;
		}
		IterBase &operator--() {
			--m_idx;
			return *this;
		}
		IterBase operator--(int) {
			IterBase tmp = *this;
			--m_idx;
			return tmp;
		}

		// Arithmetic operators
		IterBase &operator+=(difference_type offset) {
			m_idx += offset;
			return *this;
		}
		IterBase operator+(difference_type offset) const {
			return IterBase(m_dq, m_idx + offset);
		}
		friend IterBase operator+(difference_type offset, const IterBase &it) {
			return (it + offset);
		}
		IterBase &operator-=(difference_type offset) {
			m_idx -= offset;
			return *this;
		}
		IterBase operator-(difference_type offset) const {
			return IterBase(m_dq, m_idx - offset);
		}
		difference_type operator-(const IterBase &other) const {
			return (static_cast<difference_type>(m_idx) -
					static_cast<difference_type>(other.m_idx));
		}

		// Comparison operators
		bool operator==(const IterBase &other) const {
			return (m_dq == other.m_dq && m_idx == other.m_idx);
		}
		bool operator!=(const IterBase &other) const {
			return !(*this == other);
		}
		bool operator<(const IterBase &other) const {
			return (m_idx < other.m_idx);
		}

		bool operator>(const IterBase &other) const {
			return (m_idx > other.m_idx);
		}
		bool operator<=(const IterBase &other) const {
			return (m_idx <= other.m_idx);
		}
		bool operator>=(const IterBase &other) const {
			return (m_idx >= other.m_idx);
		}
	};

public:
	// Standard container typedefs
	using iterator = IterBase<false>;
	using const_iterator = IterBase<true>;
	using reverse_iterator = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	// Constructors, Destructor, and Assignment
	Deque() noexcept : m_data(nullptr), m_head(0), m_sz(0), m_cap(0) {}
	explicit Deque(size_type cnt)
		: m_data(nullptr), m_head(0), m_sz(0), m_cap(0) {
		if (cnt > 0) {
			m_initElems(cnt);
			for (size_type i = 0; i < cnt; ++i) new (m_data + i) T();
			m_sz = cnt;
		}
	}
	Deque(size_type cnt, const T &val)
		: m_data(nullptr), m_head(0), m_sz(0), m_cap(0) {
		m_initElems(cnt);
		for (size_type i = 0; i < cnt; ++i) new (m_data + i) T(val);
		m_sz = cnt;
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	Deque(Iter first, Iter last) : Deque() { assign(first, last); }
	Deque(const Deque &other) : Deque() {
		if (other.empty()) return;
		m_initElems(other.m_sz);
		for (size_type i = 0; i < other.m_sz; ++i) {
			new (m_data + i) T(other[i]);
		}
		m_sz = other.m_sz;
	}
	Deque(Deque &&other) noexcept
		: m_data(other.m_data), m_head(other.m_head),
		  m_sz(other.m_sz), m_cap(other.m_cap) {
		other.m_data = nullptr;
		other.m_head = 0;
		other.m_sz = 0;
		other.m_cap = 0;
	}
	Deque(std::initializer_list<T> ilist) : Deque(ilist.begin(), ilist.end()) {}

	~Deque() {
		m_destroyAllElems();
		::operator delete(m_data);
	}

	Deque &operator=(const Deque &other) {
		if (this != &other) {
			Deque tmp(other);
			swap(tmp);
		}
		return *this;
	}

	Deque &operator=(Deque &&other) noexcept {
		if (this != &other) {
			m_destroyAllElems();
			::operator delete(m_data);

			m_data = other.m_data;
			m_head = other.m_head;
			m_sz = other.m_sz;
			m_cap = other.m_cap;

			other.m_data = nullptr;
			other.m_head = 0;
			other.m_sz = 0;
			other.m_cap = 0;
		}
		return *this;
	}

	Deque &operator=(std::initializer_list<T> ilist) {
		assign(ilist.begin(), ilist.end());
		return *this;
	}

	void assign(size_type cnt, const T &val) {
		clear();
		if (cnt > m_cap) m_reallocate(cnt);
		for (size_type i = 0; i < cnt; ++i) new (m_data + i) T(val);
		m_sz = cnt;
		m_head = 0;
	}

	template <typename Iter, typename = RequireInputIter<Iter>>
	void assign(Iter first, Iter last) {
		clear();
		for (; first != last; ++first) pushBack(*first);
	}

	// Element Access

	reference at(size_type pos) {
		m_rangeCheck(pos);
		return (*this)[pos];
	}
	const_reference at(size_type pos) const {
		m_rangeCheck(pos);
		return (*this)[pos];
	}
	reference operator[](size_type pos) {
		return m_data[(m_head + pos) % m_cap];
	}
	const_reference operator[](size_type pos) const {
		return m_data[(m_head + pos) % m_cap];
	}

	reference front() { return m_data[m_head]; }
	const_reference front() const { return m_data[m_head]; }

	reference back() { return m_data[(m_head + m_sz - 1) % m_cap]; }
	const_reference back() const { return m_data[(m_head + m_sz - 1) % m_cap]; }

	// Iterators

	iterator begin() noexcept { return iterator(this, 0); }
	const_iterator begin() const noexcept { return const_iterator(this, 0); }
	const_iterator cbegin() const noexcept { return const_iterator(this, 0); }

	iterator end() noexcept { return iterator(this, m_sz); }
	const_iterator end() const noexcept { return const_iterator(this, m_sz); }
	const_iterator cend() const noexcept { return const_iterator(this, m_sz); }

	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	const_reverse_iterator rbegin() const noexcept {
		return const_reverse_iterator(end());
	}
	const_reverse_iterator crbegin() const noexcept {
		return const_reverse_iterator(cend());
	}

	reverse_iterator rend() noexcept {
		return reverse_iterator(begin());
	}
	const_reverse_iterator rend() const noexcept {
		return const_reverse_iterator(begin());
	}
	const_reverse_iterator crend() const noexcept {
		return const_reverse_iterator(cbegin());
	}

	// Capacity
	bool empty() const noexcept { return !m_sz; }
	size_type size() const noexcept { return m_sz; }
	size_type capacity() const noexcept { return m_cap; }
	void shrinkToFit() {
		if (m_sz < m_cap) m_reallocate(m_sz);
	}

	// Modifiers
	void clear() noexcept {
		m_destroyAllElems();
		m_head = 0;
		m_sz = 0;
	}

	iterator insert(const_iterator pos, const T &val) {
		return emplace(pos, val);
	}
	iterator insert(const_iterator pos, T &&val) {
		return emplace(pos, std::move(val));
	}
	iterator insert(const_iterator pos, size_type cnt, const T &val) {
		const difference_type idx = pos - cbegin();
		if (!cnt) return (begin() + idx);
		if (m_sz + cnt > m_cap) {
			const size_type new_cap = std::max(m_cap * 2, m_sz + cnt);
			m_reallocateAndInsert(new_cap, idx, cnt);
		} else {
			const size_type elems_after = m_sz - idx;
			// Shift the smaller part of the deque
			if (idx < elems_after) {
				// Shift front part backward
				const size_type old_head = m_head;
				m_head = ((m_head >= cnt) ? (m_head - cnt) : (m_cap - (cnt - m_head)));
				for (size_type i = 0; i < idx; ++i) {
					(*this)[i] = std::move_if_noexcept(m_data[(old_head + i) % m_cap]);
				}
			} else {
				// Shift back part forward
				for (size_type i = m_sz - 1; i >= idx; --i) {
					(*this)[i + cnt] = std::move_if_noexcept((*this)[i]);
				}
			}
		}
		for (size_type i = 0; i < cnt; ++i) {
			m_data[(m_head + idx + i) % m_cap] = val;
		}
		m_sz += cnt;
		return (begin() + idx);
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	iterator insert(const_iterator pos, Iter first, Iter last) {
		const difference_type idx = pos - cbegin();
		Deque tmp(first, last); // Easiest way to handle all iterator types
		const size_type cnt = tmp.size();
		if (cnt == 0) return (begin() + idx);
		if (m_sz + cnt > m_cap) {
			m_reallocateAndInsert(std::max(m_cap * 2, m_sz + cnt), idx, cnt);
		} else {
			const size_type elems_after = m_sz - idx;
			if (idx < elems_after) {
				const size_type old_head = m_head;
				m_head = (m_head >= cnt) ? m_head - cnt : m_cap - (cnt - m_head);
				for (size_type i = 0; i < idx; ++i)
					(*this)[i] = std::move_if_noexcept(m_data[(old_head + i) % m_cap]);
			} else {
				for (size_type i = m_sz - 1; i >= idx; --i)
					(*this)[i + cnt] = std::move_if_noexcept((*this)[i]);
			}
		}
		for (size_type i = 0; i < cnt; ++i) {
			(*this)[idx + i] = std::move_if_noexcept(tmp[i]);
		}
		m_sz += cnt;
		return (begin() + idx);
	}
	template <class... Args>
	iterator emplace(const_iterator pos, Args &&...args) {
		const difference_type idx = pos - cbegin();
		// If inserting at front or back, delegate to existing efficient functions
		if (idx == 0) {
			emplaceFront(std::forward<Args>(args)...);
			return begin();
		}
		if (static_cast<size_type>(idx) == m_sz) {
			emplaceBack(std::forward<Args>(args)...);
			return (end() - 1);
		}
		m_expandIfFull();
		// To insert in the middle, it's cheaper to move the smaller half.
		if (idx < m_sz / 2) {
			// Shift elements [0...idx-1] one step to the left
			pushFront(std::move(front()));
			for (size_type i = 1; i < idx; ++i) {
				(*this)[i] = std::move((*this)[i + 1]);
			}
		} else {
			// Shift elements [idx...size-1] one step to the right
			pushBack(std::move(back()));
			for (size_type i = m_sz - 2; i > idx; --i) {
				(*this)[i] = std::move((*this)[i - 1]);
			}
		}
		(*this)[idx] = T(std::forward<Args>(args)...);
		return (begin() + idx);
	}

	iterator erase(const_iterator pos) { return erase(pos, pos + 1); }
	iterator erase(const_iterator first, const_iterator last) {
		const difference_type idx_first = first - cbegin();
		const difference_type idx_last = last - cbegin();
		const difference_type cnt = idx_last - idx_first;
		if (cnt <= 0) return (begin() + idx_first);
		// Decide which part of the deque to shift
		if (idx_first < (m_sz - idx_last)) {
			// The front part is smaller, so move [0..idx_first-1] to the right
			std::move_backward(begin(), begin() + idx_first, begin() + idx_last);
			for (size_type i = 0; i < cnt; ++i) {
				popFront();
			}
		} else {
			// The back part is smaller, so move [idx_last..end] to the left
			std::move(begin() + idx_last, end(), begin() + idx_first);
			for (size_type i = 0; i < cnt; ++i) {
				popBack();
			}
		}
		return (begin() + idx_first);
	}

	template <class... Args>
	void emplaceFront(Args &&...args) {
		m_expandIfFull();
		m_head = (m_head == 0) ? m_cap - 1 : m_head - 1;
		new (m_data + m_head) T(std::forward<Args>(args)...);
		++m_sz;
	}
	void pushFront(const T &val) { emplaceFront(val); }
	void pushFront(T &&val) { emplaceFront(std::move(val)); }
	void popFront() {
		m_data[m_head].~T();
		m_head = (m_head + 1) % m_cap;
		--m_sz;
	}

	template <class... Args>
	void emplaceBack(Args &&...args) {
		m_expandIfFull();
		size_type tail_pos = (m_head + m_sz) % m_cap;
		new (m_data + tail_pos) T(std::forward<Args>(args)...);
		++m_sz;
	}
	void pushBack(const T &val) { emplaceBack(val); }
	void pushBack(T &&val) { emplaceBack(std::move(val)); }
	void popBack() {
		size_type last_pos = (m_head + m_sz - 1) % m_cap;
		m_data[last_pos].~T();
		--m_sz;
	}

	void resize(size_type cnt) {
		if (cnt < m_sz) {
			for (size_type i = cnt; i < m_sz; ++i) (*this)[i].~T();
		} else if (cnt > m_sz) {
			if (cnt > m_cap) m_reallocate(cnt);
			size_type old_size = m_sz;
			for (size_type i = old_size; i < cnt; ++i) emplaceBack();
		}
		m_sz = cnt;
	}

	void resize(size_type cnt, const value_type &val) {
		if (cnt < m_sz) {
			for (size_type i = cnt; i < m_sz; ++i) (*this)[i].~T();
		} else if (cnt > m_sz) {
			if (cnt > m_cap) m_reallocate(cnt);
			size_type old_size = m_sz;
			for (size_type i = old_size; i < cnt; ++i) pushBack(val);
		}
		m_sz = cnt;
	}

	void swap(Deque &other) noexcept {
		using std::swap;
		swap(m_data, other.m_data);
		swap(m_head, other.m_head);
		swap(m_sz, other.m_sz);
		swap(m_cap, other.m_cap);
	}

private:
	T *m_data;
	size_type m_head;
	size_type m_sz;
	size_type m_cap;

	static const size_type INITIAL_CAPACITY = 8;

	void m_rangeCheck(size_type pos) const {
		if (pos >= m_sz) {
			throw std::out_of_range("Deque::m_rangeCheck: pos (which is " +
									std::to_string(pos) +
									") >= this->size() (which is " +
									std::to_string(m_sz) + ")");
		}
	}

	void m_reallocate(size_type new_cap) {
		if (new_cap < m_sz) return;

		T *new_data = static_cast<T *>(::operator new(new_cap * sizeof(T)));
		size_type elems_moved = 0;
		try {
			for (size_type i = 0; i < m_sz; ++i) {
				// Use move-if-noexcept for performance and exception safety
				new (new_data + i) T(std::move_if_noexcept((*this)[i]));
				elems_moved++;
			}
		} catch (...) {
			for (size_type i = 0; i < elems_moved; ++i) (new_data + i)->~T();
			::operator delete(new_data);
			throw;
		}

		m_destroyAllElems();
		::operator delete(m_data);

		m_data = new_data;
		m_head = 0;
		m_cap = new_cap;
	}
	void m_reallocateAndInsert(size_type new_cap, size_type gap_idx, size_type gap_cnt) {
		T *new_data = static_cast<T *>(::operator new(new_cap * sizeof(T)));
		size_type elems_moved = 0;
		try {
			// Move elements before the gap
			for (size_type i = 0; i < gap_idx; ++i) {
				new (new_data + i) T(std::move_if_noexcept((*this)[i]));
				elems_moved++;
			}
			// Move elements after the gap
			for (size_type i = gap_idx; i < m_sz; ++i) {
				new (new_data + i + gap_cnt) T(std::move_if_noexcept((*this)[i]));
				elems_moved++;
			}
		} catch (...) {
			// Exception safety
			for (size_type i = 0; i < gap_idx; ++i) (new_data + i)->~T();
			for (size_type i = gap_idx; i < elems_moved; ++i) {
				(new_data + i + gap_cnt)->~T();
			}
			::operator delete(new_data);
			throw;
		}

		m_destroyAllElems();
		::operator delete(m_data);

		m_data = new_data;
		m_head = 0;
		m_cap = new_cap;
		// Size will be updated by the calling insert function
	}

	void m_expandIfFull() {
		if (m_sz == m_cap) {
			m_reallocate((m_cap > 0) ? m_cap * 2 : INITIAL_CAPACITY);
		}
	}
	void m_destroyAllElems() noexcept {
		for (size_type i = 0; i < m_sz; ++i) (*this)[i].~T();
	}
	void m_initElems(size_type cnt) {
		m_cap = cnt;
		m_data = static_cast<T *>(::operator new(m_cap * sizeof(T)));
		m_head = 0;
	}
};

template <typename T, typename Cmp = std::less<T>, typename Dq = Deque<T>>
class MonoDeque {
public:
	inline MonoDeque(const Cmp &cmp = Cmp()) : m_cmp(cmp) {}

	inline size_t size() const noexcept { return m_dq.size(); }
	inline bool empty() const noexcept { return m_dq.empty(); }

	inline const T &front() const { return m_dq.front(); }
	inline const T &back() const { return m_dq.back(); }

	inline void pushFront(const T &val) {
		popFrontFor(val);
		m_dq.pushFront(val);
	}
	inline void pushFront(T &&val) {
		popFrontFor(val);
		m_dq.pushFront(std::move(val));
	}
	template <typename... Args>
	inline void emplaceFront(Args &&...args) {
		auto t = T(std::forward(args)...);
		popFrontFor(t);
		m_dq.pushFront(std::move(t));
	}

	inline void pushBack(const T &val) {
		popBackFor(val);
		m_dq.pushBack(val);
	}
	inline void pushBack(T &&val) {
		popBackFor(val);
		m_dq.pushBack(std::move(val));
	}
	template <typename... Args>
	inline void emplaceBack(Args &&...args) {
		auto t = T(std::forward<Args>(args)...);
		popBackFor(t);
		m_dq.pushBack(std::move(t));
	}

	inline void popFront() { m_dq.popFront(); }
	inline void popFrontFor(const T &val) {
		while (m_dq.size() && !m_cmp(val, m_dq.front())) popFront();
	}

	inline void popBack() { m_dq.popBack(); }
	inline void popBackFor(const T &val) {
		while (size() && !m_cmp(m_dq.back(), val)) popBack();
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

template <typename KeyElem>
struct unordered_trie_set {
	struct node_type;

	using size_type = std::size_t;
	using key_element_type = KeyElem;
	using container_type = std::unordered_map<KeyElem, std::unique_ptr<node_type>>;
	struct node_type {
		bool exist;
		container_type children;
	};

	node_type root;

	unordered_trie_set() : root() {}
	unordered_trie_set(const unordered_trie_set &) = default;
	unordered_trie_set(unordered_trie_set &&) = default;
	unordered_trie_set &operator=(const unordered_trie_set &) = default;
	unordered_trie_set &operator=(unordered_trie_set &&) = default;
	~unordered_trie_set() = default;

	template <typename Iter, typename = RequireInputIter<Iter>>
	inline bool has_prefix(Iter begin, Iter end) const {
		const node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto j = p->children.find(*i);
			if (j == p->children.end()) {
				return false;
			}
			p = j->second.get();
		}
		return true;
	}
	template <typename Container>
	inline bool has_prefix(const Container &key) const {
		return has_prefix(key.begin(), key.end());
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline size_type count(Iter begin, Iter end) const {
		const node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto j = p->children.find(*i);
			if (j == p->children.end()) {
				return 0;
			}
			p = j->second.get();
		}
		return p->exist;
	}
	template <typename Container>
	inline size_type count(const Container &key) const {
		return count(key.begin(), key.end());
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline void insert(Iter begin, Iter end) {
		node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto &ch = p->children[*i];
			if (!ch) {
				ch = std::make_unique<node_type>();
			}
			p = ch.get();
		}
		p->exist = true;
	}
	template <typename Container>
	inline void insert(const Container &key) {
		insert(key.begin(), key.end());
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline size_type erase(Iter begin, Iter end) {
		node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto j = p->children.find(*i);
			if (j == p->children.end()) {
				return 0;
			}
			p = j->second.get();
		}
		if (p->exist) {
			p->exist = false;
			return 1;
		}
		return 0;
	}
	template <typename Container>
	inline size_type erase(const Container &key) {
		return erase(key.begin(), key.end());
	}
};

template <typename KeyElem>
struct unordered_trie_multiset {
	struct node_type;

	using size_type = std::size_t;
	using key_element_type = KeyElem;
	using container_type = std::unordered_map<KeyElem, std::unique_ptr<node_type>>;
	struct node_type {
		size_type count;
		container_type children;
	};

	node_type root;

	unordered_trie_multiset() = default;
	unordered_trie_multiset(const unordered_trie_multiset &) = default;
	unordered_trie_multiset(unordered_trie_multiset &&) = default;
	unordered_trie_multiset &operator=(const unordered_trie_multiset &) = default;
	unordered_trie_multiset &operator=(unordered_trie_multiset &&) = default;
	~unordered_trie_multiset() = default;

	template <typename Iter, typename = RequireInputIter<Iter>>
	inline size_type has_prefix(Iter begin, Iter end) const {
		const node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto j = p->children.find(*i);
			if (j == p->children.end()) {
				return false;
			}
			p = j->second.get();
		}
		return true;
	}
	template <typename Container>
	inline size_type has_prefix(const Container &key) const {
		return has_prefix(key.begin(), key.end());
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline size_type count(Iter begin, Iter end) const {
		const node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto j = p->children.find(*i);
			if (j == p->children.end()) {
				return 0;
			}
			p = j->second.get();
		}
		return p->count;
	}
	template <typename Container>
	inline size_type count(const Container &key) const {
		return count(key.begin(), key.end());
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline void insert(Iter begin, Iter end) {
		node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto &ch = p->children[*i];
			if (!ch) {
				ch = std::make_unique<node_type>();
			}
			p = ch.get();
		}
		++(p->count);
	}
	template <typename Container>
	inline void insert(const Container &key) {
		insert(key.begin(), key.end());
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline size_type erase_one(Iter begin, Iter end) {
		node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto j = p->children.find(*i);
			if (j == p->children.end()) {
				return 0;
			}
			p = j->second.get();
		}
		if (p->count) {
			--(p->count);
			return 1;
		}
		return 0;
	}
	template <typename Container>
	inline size_type erase_one(const Container &key) {
		return erase_one(key.begin(), key.end());
	}
	template <typename Iter, typename = RequireInputIter<Iter>>
	inline size_type erase(Iter begin, Iter end) {
		node_type *p = &root;
		for (auto i = begin; i != end; ++i) {
			auto j = p->children.find(*i);
			if (j == p->children.end()) {
				return 0;
			}
			p = j->second.get();
		}
		size_type ret = p->count;
		p->count = 0;
		return ret;
	}
	template <typename Container>
	inline size_type erase(const Container &key) {
		return erase(key.begin(), key.end());
	}
};

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
		if (n >= m_tree.size()) n = m_tree.size() - 1;
		T res = id_elem;
		for (; n; n -= lowbit(n)) res = m_oper(res, m_tree[n]);
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
			if (j < m_tree.size()) m_tree[j] = m_oper(m_tree[j], m_tree[i]);
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

/**
 * @todo Make it a template!
 */
class SegTree {
public:
	SegTree(size_t sz) : m_sz(sz) {}

	inline void modify(size_t pos, size_t len, int64_t diff) {
		m_modify(&m_rt, 0, m_sz, pos, pos + len, diff);
	}

	inline int64_t query(size_t pos, size_t len) {
		return m_query(&m_rt, 0, m_sz, pos, pos + len);
	}

protected:
	struct Node {
		int64_t val = 0, lzy = 0;
		Node *lch = nullptr, *rch = nullptr;

		inline void pushDown() {
			if (!lch) {
				lch = new Node({.lzy = lzy});
			} else {
				lch->lzy += lzy;
			}
			if (!rch) {
				rch = new Node({.lzy = lzy});
			} else {
				rch->lzy += lzy;
			}
			lzy = 0;
		}

		~Node() {
			if (lch) delete lch;
			if (rch) delete rch;
		}
	};

	Node m_rt;
	size_t m_sz;

	void m_modify(Node *p, size_t nd_beg, size_t nd_end, size_t beg, size_t end, int64_t diff) {
		assert(nd_beg <= beg && beg < end && end <= nd_end);
		p->val += (end - beg) * diff;
		if (nd_beg == beg && nd_end == end) {
			p->lzy += diff;
			return;
		}
		p->pushDown();
		size_t nd_mid = nd_beg + ((nd_end - nd_beg) >> 1);
		if (beg < nd_mid) {
			m_modify(p->lch, nd_beg, nd_mid, beg, std::min(end, nd_mid), diff);
		}
		if (end > nd_mid) {
			m_modify(p->rch, nd_mid, nd_end, std::max(beg, nd_mid), end, diff);
		}
	}

	int64_t m_query(Node *p, size_t nd_beg, size_t nd_end, size_t beg, size_t end) {
		assert(nd_beg <= beg && beg < end && end <= nd_end);
		if (nd_beg == beg && nd_end == end) return p->val;
		int64_t res = (end - beg) * p->lzy;
		size_t nd_mid = nd_beg + ((nd_end - nd_beg) >> 1);
		if (beg < nd_mid && p->lch) {
			res += m_query(p->lch, nd_beg, nd_mid, beg, std::min(end, nd_mid));
		}
		if (end > nd_mid && p->rch) {
			res += m_query(p->rch, nd_mid, nd_end, std::max(beg, nd_mid), end);
		}
		return res;
	}

private:
};

/**
 * @return sorted unique elements
 */
template <typename Iter,
		  typename Cmp = std::less<typename std::iterator_traits<Iter>::value_type>,
		  typename Eq = std::equal_to<typename std::iterator_traits<Iter>::value_type>>
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

/**
 * @return indices of elements consisting
 * 	one longest increasing subsequence of [@param begin, @param end)
 */
template <typename Iter,
		  typename Cmp = std::less<typename std::iterator_traits<Iter>::value_type>,
		  typename = RequireFwdIter<Iter>>
std::vector<size_t> longestIncrSubseq(Iter begin, Iter end, Cmp cmp = Cmp()) {
	auto cmpVal = [begin, end, &cmp](size_t lhs, size_t rhs) {
		return cmp(*std::next(begin, lhs), *std::next(begin, rhs));
	};
	auto pre = std::vector<size_t>(std::distance(begin, end));
	std::vector<size_t> min_end, res;
	min_end.reserve(pre.size());
	for (size_t i = 0; begin != end; ++begin, ++i) {
		size_t j = std::lower_bound(min_end.begin(), min_end.end(), i, cmpVal) -
				   min_end.begin();
		if (j == min_end.size()) {
			min_end.emplace_back(i);
		} else {
			min_end[j] = i;
		}
		pre[i] = (j ? min_end[j - 1] : size_t(-1));
	}
	res.reserve(min_end.size());
	for (size_t p = min_end.back(); ~p; p = pre[p]) {
		res.emplace_back(p);
	}
	std::reverse(res.begin(), res.end());
	return res;
}

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

/**
 * @return the number of inversions
 */
template <typename OutputIter, typename AuxIter>
size_t mergeSort(OutputIter first, OutputIter last, AuxIter aux) {
	size_t n = std::distance(first, last);
	if (n <= 1) return 0;
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
	while (i != mid) *(k++) = *(i++);
	while (j != last) *(k++) = *(j++);
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

template <typename Uint = uint64_t, typename Aux = __uint128_t,
		  typename = std::enable_if_t<std::is_unsigned<Uint>::value &&
									  std::is_unsigned<Aux>::value &&
									  sizeof(Uint) * 2 <= sizeof(Aux)>>
inline Uint qPowMod(Uint base, Uint exp, Uint mod) {
	Aux mul = base, res = 1;
	while (exp) {
		if (exp & 1) res = res * mul % mod;
		mul = mul * mul % mod;
		exp >>= 1;
	}
	return res;
}

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
	if (arr.empty()) return arr;
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

std::vector<uint32_t> sieveOfEratosthenes(uint32_t max) {
	if (max <= 1) return {};
	auto not_prime = std::vector<bool>(max + 1);
	std::vector<uint32_t> primes;
	primes.emplace_back(2);
	for (uint32_t i = 4; i <= max; i += 2) {
		not_prime[i] = true;
	}
	for (uint32_t i = 3; i * i <= max; i += 2) {
		if (not_prime[i]) continue;
		for (uint32_t j = (i << 1); j <= max; j += i) not_prime[j] = true;
	}
	for (uint32_t i = 3; i <= max; i += 2) {
		if (!not_prime[i]) primes.emplace_back(i);
	}
	return primes;
}

std::vector<uint32_t> linearSieve(uint32_t max) {
	if (max <= 1) return {};
	auto not_prime = std::vector<bool>(max + 1);
	std::vector<uint32_t> primes;
	primes.emplace_back(2);
	for (uint32_t i = 3; i <= max; i += 2) {
		if (!not_prime[i]) primes.emplace_back(i);
		for (auto j : primes) {
			if (j * i > max) break;
			not_prime[j * i] = true; // j is the minimum prime factor of i * j
			if (i % j == 0) break;
		}
	}
	return primes;
}

/**
 * The Euler's totient function phi(n) is defined as the number of integers
 * 	that are less than or equal to n and coprime to n.
 * @brief Calculate phi(n) using the formula phi(n) = n * prod(1 - 1 / p_i)
 * @return phi(n)
 */
uint64_t eulerFuncOf(uint64_t n) {
	uint64_t res = n;
	for (uint64_t i = 2; i * i <= n; ++i) {
		if (n % i == 0) {
			res = res / i * (i - 1);
			do n /= i;
			while (n % i == 0);
		}
	}
	if (n > 1) res = res / n * (n - 1);
	return res;
}

/**
 * @return phi[0, max]
 */
std::vector<uint32_t> eulerFuncsOf(size_t max) {
	if (max == 0) return {0};
	if (max == 1) return {0, 1};

	std::vector<bool> not_prime(max + 1);
	std::vector<uint32_t> primes;
	primes.reserve(3 * max / (floorLogn2(max) << 1)); // approximate number of primes
	std::vector<uint32_t> phi(max + 1);

	phi[1] = 1;
	for (uint32_t i = 2; i <= max; ++i) {
		if (!not_prime[i]) {
			primes.emplace_back(i);
			phi[i] = i - 1;
		}
		for (auto prime : primes) {
			if (i * prime > max) break;
			not_prime[i * prime] = true;
			if (i % prime == 0) {
				phi[i * prime] = phi[i] * prime;
				break;
			}
			phi[i * prime] = phi[i] * (prime - 1);
		}
	}
	return phi;
}

/**
 * @return @param num is prime or not
 * @note time complexity O(k * log^3(num))
 */
bool millerRabin(uint64_t num) {
	if (num < 2) return false;
	if (num == 2 || num == 3) return true;
	if (!(num & 1)) return false;

	uint64_t d = num - 1;
	unsigned s = __builtin_ctzll(d);
	d >>= s;

	constexpr std::array<uint64_t, 12>
		BASES{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
	for (auto base : BASES) {
		if (num <= base) break;

		uint64_t x = qPowMod(base, d, num);
		if (x == 1 || x == num - 1) continue;

		bool is_composite = true;
		for (unsigned r = 1; r < s; ++r) {
			x = static_cast<__uint128_t>(x) * x % num;
			if (x == num - 1) {
				is_composite = false;
				break;
			}
		}

		if (is_composite) return false;
	}
	return true;
}

/**
 * @return a non-trivial factor of @param num
 * @note expected time complexity O(n^(1/4))
 */
uint64_t pollardRho(uint64_t num) {
	if (!(num & 1)) return 2;
	if (millerRabin(num)) return num;

	std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<uint64_t> distrib(1, num - 1);

	uint64_t y = distrib(rng), c = distrib(rng), m = distrib(rng),
			 g = 1, r = 1, q = 1, x = 0, ys = 0;

	while (g == 1) {
		x = y;
		for (uint64_t i = 0; i < r; ++i) {
			y = (static_cast<__uint128_t>(y) * y % num + c) % num;
		}

		uint64_t k = 0;
		while (k < r && g == 1) {
			ys = y;
			for (int i = 0; i < std::min(uint64_t(128), r - k); i++) {
				y = (static_cast<__uint128_t>(y) * y % num + c) % num;
				// q = q * |x - y| % num
				uint64_t diff = (x > y) ? (x - y) : (y - x);
				q = static_cast<__uint128_t>(q) * diff % num;
			}
			g = std::__gcd(q, num);
			k += 128;
		}
		r <<= 1;
	}

	if (g == num) {
		while (true) {
			ys = (static_cast<__uint128_t>(ys) * ys % num + c) % num;
			uint64_t diff = (x > ys) ? (x - ys) : (ys - x);
			g = std::__gcd(diff, num);
			if (g > 1) break;
		}
	}

	return g;
}

void factorize(uint64_t num, std::map<uint64_t, uint32_t> &factors) {
	if (num == 1) return;
	if (millerRabin(num)) {
		++factors[num];
		return;
	}

	uint64_t factor = num;
	while (factor == num) factor = pollardRho(num);
	factorize(factor, factors);
	factorize(num / factor, factors);
}

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
		if (i & 1) rev[i] |= (n >> 1);
	}
	Iter iter = begin;
	for (size_t i = 0; i != n; ++i, ++iter) {
		if (i < rev[i]) std::swap(*iter, *std::next(begin, rev[i]));
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
		for (auto &x : arr) x /= arr.size();
	}
}

template <typename Iter, typename = RequireFwdIter<Iter>>
inline std::vector<size_t> prefFuncOf(Iter begin, Iter end) {
	auto pi = std::vector<size_t>(std::distance(begin, end));
	end = std::next(begin);
	for (size_t i = 1, j; i < pi.size(); ++i, ++end) {
		for (j = pi[i - 1]; j && *end != *std::next(begin, j); j = pi[j - 1]);
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
		if (pi[i] == pattern_sz) res.emplace_back(i - (pattern_sz << 1));
	}
	return res;
}

template <typename Iter>
inline std::vector<size_t> borderLengths(Iter begin, Iter end) {
	auto pi = prefFuncOf(begin, end);
	std::vector<size_t> res;
	for (size_t i = pi.size(); i; i = pi[i - 1]) res.emplace_back(pi[i - 1]);
	return res;
}

/**
 * @note the last period may be incomplete
 */
template <typename Iter>
inline size_t periodLength(Iter begin, Iter end) {
	if (begin == end) return 0;
	auto pi = prefFuncOf(begin, end);
	return (pi.size() - pi.back());
}

template <typename Mod = uint32_t,
		  typename Aux = uint64_t,
		  typename = std::enable_if_t<std::is_unsigned<Mod>::value &&
									  std::is_unsigned<Aux>::value &&
									  sizeof(Mod) * 2 == sizeof(Aux)>>
std::vector<Mod> hashOf(const std::string &s,
						Mod base = 233, Mod mod = 993244853) {
	auto h = std::vector<Mod>(s.size() + 1);
	Aux b = base;
	for (size_t i = 0; i != s.size(); ++i) h[i + 1] = (h[i] * b + s[i]) % mod;
	return h;
}

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

	for (size_t i = 0; i < n; ++i) ++cnt[ra[i]];
	for (size_t i = 1; i < unq; ++i) cnt[i] += cnt[i - 1];
	for (size_t i = n; (i--) > 0;) sa[--cnt[ra[i]]] = i;

	for (size_t len = 1, tot; len < n; len <<= 1) {
		// Sort sa so that ra[sa[i] + len] <= ra[sa[i + 1] + len]
		tot = 0;
		for (size_t i = n - len; i != n; ++i) tmp[tot++] = i;
		for (size_t i = 0; i != n; ++i) {
			if (sa[i] >= len) tmp[tot++] = sa[i] - len;
		}

		// Stably sort sa so that ra[sa[i]] <= ra[sa[i + 1]]
		std::fill(cnt.begin(), cnt.begin() + unq, 0);
		for (auto i : tmp) ++cnt[ra[i]];
		for (size_t i = 1; i != unq; ++i) cnt[i] += cnt[i - 1];
		for (size_t i = n; (i--) > 0;) sa[--cnt[ra[tmp[i]]]] = tmp[i];

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
		if (len) --len;
		for (j = sa[ra[i] - 1];
			 *std::next(begin, i + len) == *std::next(begin, j + len);
			 ++len);
		height[ra[i]] = len;
	}

	return {sa, ra, height};
}

/**
 * @return {odd, even}, both denote the number of palindrome subsequences
 * 	centering around each elements (consider the right one as the center for
 * 	even-length palindromes)
 */
template <typename Iter, typename = RequireFwdIter<Iter>>
inline std::pair<std::vector<size_t>, std::vector<size_t>>
manacher(Iter begin, Iter end) {
	size_t n = std::distance(begin, end);
	if (!n) return {{}, {}};
	auto odd = std::vector<size_t>(n), even = std::vector<size_t>(n);
	odd[0] = 1;
	for (size_t i = 1, b = 0, e = 1; i != n; ++i) {
		if (i < e) odd[i] = std::min(odd[b + e - 1 - i], e - i);
		while (odd[i] <= i && i + odd[i] < n &&
			   *std::next(begin, i - odd[i]) ==
				   *std::next(begin, i + odd[i])) ++odd[i];
		if (i + odd[i] > e) {
			b = i - odd[i] + 1;
			e = i + odd[i];
		}
	}
	for (size_t i = 0, b = 0, e = 0; i != n; ++i) {
		if (i < e) even[i] = std::min(even[b + e - i], e - i);
		while (even[i] < i && i + even[i] < n &&
			   *std::next(begin, i - even[i] - 1) ==
				   *std::next(begin, i + even[i])) ++even[i];
		if (i + even[i] > e) {
			b = i - even[i];
			e = i + even[i];
		}
	}
	return {odd, even};
}

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
		for (auto &lst : m_adj) lst.clear();
	}
	void assign(size_t n, const std::vector<Edge> &edges) {
		m_edges = edges;
		m_adj.resize(n);
		for (auto &lst : m_adj) lst.clear();
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

	std::pair<Weight, std::vector<bool>>
	dinic(size_t src, size_t dst,
		  Weight lim = std::numeric_limits<Weight>::max()) const;

protected:
	std::vector<Edge> m_edges;
	std::vector<std::vector<size_t>> m_adj;

	void m_insertToAdj(size_t idx) {
		size_t sz_requirement = std::max(m_edges[idx].u, m_edges[idx].v) + 1;
		if (sz_requirement > m_adj.size()) m_adj.resize(sz_requirement);
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
		if (dfn[i] == size_t(-1)) dfs(i);
	}
	return sccs;
}

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
			if (vbcc_stk.size()) vbccs.emplace_back(std::move(vbcc_stk));
			if (ebcc_stk.size()) ebccs.emplace_back(std::move(ebcc_stk));
		}
	}
	return {{cut_verts, vbccs}, {bridges, ebccs}};
}

template <typename Weight, bool is_directed>
inline std::vector<size_t> Graph<Weight, is_directed>::toposort() const {
	static_assert(is_directed,
				  "Topological sorting is only applicable to directed graphs.");
	std::vector<size_t> indeg(m_adj.size());
	for (auto &edge : m_edges) ++indeg[edge.v];
	std::vector<size_t> res, zero_indeg_verts;
	for (size_t i = 0; i != m_adj.size(); ++i) {
		if (!indeg[i]) zero_indeg_verts.emplace_back(i);
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

template <typename Weight, bool is_directed>
inline std::vector<std::vector<size_t>>
Graph<Weight, is_directed>::kruskal() const {
	static_assert(!is_directed,
				  "Kruskal's algorithm is only applicable to undirected graphs.");
	if (m_adj.size() < 2) return {};

	std::vector<size_t> sorted(m_edges.size());
	std::iota(sorted.begin(), sorted.end(), 0);
	std::sort(sorted.begin(), sorted.end(),
			  [&](size_t lhs, size_t rhs) {
				  return (m_edges[lhs].w < m_edges[rhs].w);
			  });

	std::vector<size_t> dsu(m_adj.size());
	std::iota(dsu.begin(), dsu.end(), 0);
	auto find = [&](size_t x) -> size_t {
		while (dsu[dsu[x]] != dsu[x]) dsu[x] = dsu[dsu[x]];
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
	for (auto i : msf_edges) components[find(m_edges[i].u)].emplace_back(i);
	std::vector<std::vector<size_t>> res;
	res.reserve(components.size());
	for (auto &pr : components) res.emplace_back(std::move(pr.second));
	return res;
}

template <typename Weight, bool is_directed>
inline std::vector<size_t> Graph<Weight, is_directed>::prim(size_t rt) const {
	static_assert(!is_directed,
				  "Prim algorithm is only applicable to undirected graphs.");
	if (m_adj.size() < 2) return {};

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
			if (!vis[(edge.u == frm ? edge.v : edge.u)]) pq.push(i);
		}
	};

	visit(rt);
	while (pq.size() && mst_edges.size() + 1 < m_adj.size()) {
		size_t i = pq.top();
		pq.pop();
		size_t u = m_edges[i].u, v = m_edges[i].v;
		if (vis[u] && vis[v]) continue;
		mst_edges.emplace_back(i);
		visit(vis[u] ? v : u);
	}

	return mst_edges;
}

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
		if (vis[frm]) continue;
		vis[frm] = true;
		for (auto i : m_adj[frm]) {
			auto to = (is_directed
						   ? m_edges[i].v
						   : ((m_edges[i].u == frm)
								  ? m_edges[i].v
								  : m_edges[i].u));
			auto w = m_edges[i].w;
			if (dist[to] > dist[frm] + w) {
				pq.emplace(dist[to] = dist[frm] + w, to);
			}
		}
	}

	return dist;
}

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
			if (!is_directed) flag &= relax(edge.v, edge.u, edge.w);
		}
		if (flag) return dist;
	}
	return {}; // negative cycle detected
}

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
				if ((cnt[to] = cnt[frm] + 1) >= m_adj.size()) return {};
				if (!inq[to]) {
					inq[to] = true;
					q.emplace(to);
				}
			}
		}
	}
	return dist;
}

template <typename Weight>
inline std::vector<std::vector<Weight>>
floyd(std::vector<std::vector<Weight>> weights) {
	size_t n = weights.size();
	if (!n) return {};
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
	if (!n) return {};
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
	for (size_t i = 0; i != n; ++i) res[i][i] = 0;
	for (; n; n >>= 1) {
		if (n & 1) res = matMul(res, weights);
		weights = matMul(weights, weights);
	}
	return res;
}

/**
 * @return the shortest Hamiltonian cycle beginning and ending at vertex 0
 */
template <typename Weight>
inline std::vector<size_t>
shortestHamiltonianCycle(const std::vector<std::vector<Weight>> &adj_mat,
						 Weight inf = std::numeric_limits<Weight>::max()) {
	size_t n = adj_mat.size();
	if (!n) return {};
	if (n == 1) return {0};
	auto dp = std::vector<std::vector<Weight>>(uint64_t(1) << n,
											   std::vector<Weight>(n, inf));
	auto pre = std::vector<std::vector<size_t>>(
		uint64_t(1) << n,
		std::vector<size_t>(n, size_t(-1)));
	dp[1][0] = 0;
	for (uint64_t set = 1; set < (uint64_t(1) << n); set += 2) {
		for (size_t cur = (set != 1); cur < n; ++cur) {
			if (!(set & (1 << cur))) continue;
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
	if (!n) return {};
	if (n == 1) return {0};
	auto dp = std::vector<std::vector<Weight>>(uint64_t(1) << n,
											   std::vector<Weight>(n, inf));
	auto pre = std::vector<std::vector<size_t>>(uint64_t(1) << n,
												std::vector<size_t>(n, size_t(-1)));
	if (src != size_t(-1)) {
		dp[uint64_t(1) << src][src] = 0;
	} else {
		for (size_t i = 0; i < n; ++i) dp[uint64_t(1) << i][i] = 0;
	}
	for (uint64_t set = 1; set < (uint64_t(1) << n); ++set) {
		for (size_t cur = 0; cur < n; ++cur) {
			if (!(set & (uint64_t(1) << cur))) continue;
			if (dp[set][cur] == inf) continue;
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
	if (last == size_t(-1)) return {};
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

/**
 * @return {max flow, min cut} from src to dst
 */
template <typename Weight, bool is_directed>
std::pair<Weight, std::vector<bool>>
Graph<Weight, is_directed>::dinic(size_t src, size_t dst, Weight lim) const {
	static_assert(is_directed, "Dinic's algorithm for max flow and min cut"
							   "is only applicable to directed graphs.");

	std::vector<Weight> caps(m_edges.size() << 1);
	std::vector<size_t> dep(m_adj.size()), iter(m_adj.size());
	std::deque<size_t> q;

	for (size_t i = 0; i < m_edges.size(); ++i) caps[i << 1] = m_edges[i].w;

	auto bfs = [&]() -> void {
		std::fill(dep.begin(), dep.end(), size_t(-1));
		dep[src] = 0;
		q.clear();
		q.push_back(src);

		while (q.size()) {
			auto frm = q.front();
			q.pop_front();
			for (size_t e : m_adj[frm]) {
				size_t to;
				if (m_edges[e].v != frm) {
					to = m_edges[e].v, e <<= 1;
				} else {
					to = m_edges[e].u, e = ((e << 1) | 1);
				}
				if (!caps[e] || ~dep[to]) continue;
				dep[to] = dep[frm] + 1;
				if (to == dst) return;
				q.push_back(to);
			}
		}
	};
	std::function<Weight(size_t, const Weight &)> dfs =
		[&](size_t cur, const Weight &up) -> Weight {
		if (cur == dst) return up;

		Weight flow = 0;
		for (auto &i = iter[cur]; i != m_adj[cur].size(); ++i) {
			size_t nxt, e = m_adj[cur][i];
			if (m_edges[e].v != cur) {
				nxt = m_edges[e].v, e = (e << 1);
			} else {
				nxt = m_edges[e].u, e = ((e << 1) | 1);
			}
			if (!caps[e] || dep[cur] >= dep[nxt]) continue;
			auto down = dfs(nxt, std::min(up - flow, caps[e]));
			if (!down) continue;
			flow += down;
			caps[e] -= down, caps[e ^ 1] += down;
			if (flow == up) return flow;
		}
		dep[cur] = m_adj.size();
		return flow;
	};

	Weight max_flow = 0;
	while (max_flow < lim) {
		bfs();
		if (dep[dst] == size_t(-1)) break;
		std::fill(iter.begin(), iter.end(), 0);
		auto cur_flow = dfs(src, lim - max_flow);
		if (!cur_flow) break;
		max_flow += cur_flow;
	}

	std::vector<bool> min_cut(m_adj.size());
	q.clear();
	q.push_back(src);
	min_cut[src] = true;
	while (q.size()) {
		auto frm = q.front();
		q.pop_front();
		for (size_t e : m_adj[frm]) {
			size_t to;
			if (m_edges[e].v != frm) {
				to = m_edges[e].v, e <<= 1;
			} else {
				to = m_edges[e].u, e = ((e << 1) | 1);
			}
			if (caps[e] && !min_cut[to]) {
				q.push_back(to);
				min_cut[to] = true;
			}
		}
	}

	return std::make_pair(max_flow, std::move(min_cut));
}