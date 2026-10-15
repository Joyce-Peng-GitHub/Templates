#import "@preview/ctheorems:1.1.3": *
#show: thmrules

#let definition = thmbox("definition", "Definition")
#let theorem = thmbox("theorem", "Theorem")
#let lemma = thmbox("lemma", "Lemma")
#let proof = thmproof("proof", "Proof of")
#let algorithm = thmbox("algorithm", "Algorithm")

#set page(margin: 1.6cm)
#set text(font: "New Computer Modern")
#let indent = 2em
#set par(first-line-indent: (amount: indent, all: true))
#set heading(numbering: "1.")
#set underline(offset: 2pt)
#set page(numbering: "1")
#show raw: set text(font: "Ubuntu Mono")

#set document(
  title: [Templates],
  author: "Joyce Peng",
  date: datetime.today(),
)

#align(center)[
  #title()

  #context { document.author.join(", ") }

  #context { document.date.display() }
]

#outline()

#pagebreak()

= Utilities

== `.clang-format` Configuration

```clang-format
{
	BasedOnStyle: LLVM,
	UseTab: Always,
	IndentWidth: 4,
	TabWidth: 4,
	BreakBeforeBraces: Attach,
	AllowShortIfStatementsOnASingleLine: true,
	AllowShortLoopsOnASingleLine: true,
	IndentCaseLabels: false,
	AccessModifierOffset: -4,
	NamespaceIndentation: All,
	FixNamespaceComments: false,
	ColumnLimit: 0
}
```

== Iterator Requirements

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

== Register `__uint128_t` as Unsigned

```cpp
namespace std {
	template <>
	struct is_unsigned<__uint128_t> : public true_type {};
}
```

== Binary Search

```cpp
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
```

== $log_2$ of Integers

```cpp
inline uint64_t lowbit(uint64_t x) { return (x & (-x)); }

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

= Optimization Tricks

== Fast I/O

```cpp
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
	if (neg) {
		while (std::isdigit(ch)) {
			res = res * 10 - (ch ^ '0');
			ch = std::fgetc(file);
		}
	} else {
		while (std::isdigit(ch)) {
			res = res * 10 + (ch ^ '0');
			ch = std::fgetc(file);
		}
	}
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
```

== Compiler Optimization Switch

```cpp
#pragma GCC optimize("O3,unroll-loops")
#ifdef __AVX2__
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#endif
```

- `avx2`: Advanced Vector Extensions 2, a set of SIMD instructions for parallel processing. This option enables the 256-bit wide SIMD instructions.
- `bmi,bmi2,lzcnt`: Bit Manipulation Instruction sets, which provide efficient bit-level operations.
- `popcnt`: Population Count instruction, which accelerates `__builtin_popcount` function family.

== Fast Modulo

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

= Data Structures

== Disjoint Set Union

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
```

== Sparse Table

```cpp
template <typename T, typename Oper>
class SparseTable {
public:
	/**
	 * @pre oper must be associative (x oper (y oper z) == (x oper y) oper z)
	 * and idempotent (x oper x == x).
	 */
	SparseTable(std::vector<T> arr, Oper oper)
		: m_oper(std::move(oper)),
		  m_table(arr.empty() ? 1 : (floorLogn2(arr.size()) + 1)) {
		m_table[0] = std::move(arr);
		m_build();
	}
	SparseTable(std::vector<T> arr) : SparseTable(std::move(arr), Oper()) {}
	SparseTable() : SparseTable(std::vector<T>()) {}
	void assign(std::vector<T> arr, Oper oper) {
		m_oper = std::move(oper);
		m_table.resize(arr.empty() ? 1 : (floorLogn2(arr.size()) + 1));
		m_table[0] = std::move(arr);
		m_build();
	}

	void clear() noexcept {
		m_table.resize(1);
		m_table[0].clear();
	}

	const std::vector<T> &data() const noexcept { return m_table[0]; }
	size_t size() const noexcept { return data().size(); }
	bool empty() const noexcept { return data().empty(); }

	const T &operator[](size_t pos) const {
		assert(pos < size());
		return data()[pos];
	}

	T query(size_t pos, size_t len) const {
		assert(len && pos + len <= size());
		size_t log_len = floorLogn2(len);
		const auto &fst = m_table[log_len][pos];
		const auto &sec = m_table[log_len][pos + len - (size_t(1) << log_len)];
		return m_oper(fst, sec);
	}

protected:
	Oper m_oper;
	std::vector<std::vector<T>> m_table; // m_table[i][j] = m_oper[j, j + 2**i)

	void m_build() {
		assert(m_table.size());
		for (size_t i = 1; i < m_table.size(); ++i) {
			m_table[i].resize(size() - (size_t(1) << i) + 1);
			for (size_t j = 0; j + (size_t(1) << i) <= size(); ++j) {
				const auto &fst = m_table[i - 1][j];
				const auto &sec = m_table[i - 1][j + (size_t(1) << (i - 1))];
				m_table[i][j] = m_oper(fst, sec);
			}
		}
	}
};
```

== Leftist Heap

A mergeable binary heap that supports insertion, merging and deletion of arbitrary nodes in $O(log n)$ time.

```cpp
template <typename T, typename Cmp = std::less<T>>
class LeftistHeap {
public:
	struct Node {
		Node *prt = nullptr;
		std::unique_ptr<Node> lch, rch;
		ptrdiff_t dist = 0;
		T val;

		Node() = default;
		template <typename U, typename... Args,
				  typename = typename std::enable_if<
					  !std::is_same<
						  typename std::decay<U>::type, Node>::value>::type>
		explicit Node(U &&u, Args &&...args)
			: val(std::forward<U>(u), std::forward<Args>(args)...) {}
		Node(const Node &other)
			: lch(other.lch ? new Node(*other.lch) : nullptr),
			  rch(other.rch ? new Node(*other.rch) : nullptr),
			  dist(other.dist), val(other.val) {
			if (lch) lch->prt = this;
			if (rch) rch->prt = this;
		}
		Node(Node &&other) = delete;
		Node &operator=(const Node &other) {
			if (this == &other) return *this;

			lch = UniquePointer(other.lch ? new Node(*other.lch) : nullptr);
			if (lch) lch->prt = this;
			rch = UniquePointer(other.rch ? new Node(*other.rch) : nullptr);
			if (rch) rch->prt = this;
			dist = other.dist;
			val = other.val;
			return *this;
		}
		Node &operator=(Node &&) = delete;
	};

	explicit LeftistHeap(Cmp cmp = Cmp()) : m_sz(0), m_cmp(cmp) {}
	explicit LeftistHeap(const T &val, Cmp cmp = Cmp())
		: m_rt(new Node(val)), m_sz(1), m_cmp(cmp) {}
	template <typename Iter, typename = RequireInputIter<Iter>>
	LeftistHeap(Iter beg, Iter end, Cmp cmp = Cmp()) : m_cmp(cmp) {
		std::queue<UniquePointer> q;
		for (; beg != end; ++beg) q.emplace(new Node(*beg));
		m_sz = q.size();
		if (q.empty()) return;

		while (q.size() > 1) {
			auto tmp = std::move(q.front());
			q.pop();
			q.emplace(m_merge(std::move(tmp), std::move(q.front())));
			q.pop();
		}

		m_rt = std::move(q.front());
		m_rt->prt = nullptr;
	}
	LeftistHeap(const LeftistHeap &other)
		: m_rt(other.m_rt ? new Node(*other.m_rt) : nullptr),
		  m_sz(other.m_sz),
		  m_cmp(other.m_cmp) {}
	LeftistHeap(LeftistHeap &&other) noexcept
		: m_rt(std::move(other.m_rt)),
		  m_sz(other.m_sz),
		  m_cmp(std::move(other.m_cmp)) {
		other.m_sz = 0;
	}
	LeftistHeap &operator=(LeftistHeap other) {
		swap(other);
		return *this;
	}

	const T &top() const noexcept {
		assert(m_rt);
		return m_rt->val;
	}
	bool empty() const noexcept { return !m_rt; }
	size_t size() const noexcept { return m_sz; }

	Node *push(T val) {
		UniquePointer new_node(new Node(std::move(val)));
		Node *p = new_node.get();
		m_rt = m_merge(std::move(m_rt), std::move(new_node));
		++m_sz;
		return p;
	}

	template <typename... Args>
	Node *emplace(Args &&...args) {
		UniquePointer new_node(new Node(std::forward<Args>(args)...));
		Node *p = new_node.get();
		m_rt = m_merge(std::move(m_rt), std::move(new_node));
		++m_sz;
		return p;
	}

	void pop() noexcept {
		assert(m_rt);
		m_rt = m_merge(std::move(m_rt->lch), std::move(m_rt->rch));
		--m_sz;
	}

	void merge(LeftistHeap &&other) noexcept {
		if (this == &other) return;
		m_sz += other.m_sz;
		other.m_sz = 0;
		m_rt = m_merge(std::move(m_rt), std::move(other.m_rt));
	}

	static LeftistHeap merge(LeftistHeap &&x, LeftistHeap &&y) noexcept {
		x.merge(std::move(y));
		return std::move(x);
	}

	void erase(Node *p) noexcept {
		assert(p);
		auto ch = m_merge(std::move(p->lch), std::move(p->rch));
		if (ch) ch->prt = p->prt;
		--m_sz;

		Node *prt = p->prt;
		if (!prt) {
			m_rt = std::move(ch);
			return;
		}
		p->prt = nullptr;
		((prt->lch.get() == p) ? prt->lch : prt->rch) = std::move(ch);
		m_pushUp(prt);
	}

	void swap(LeftistHeap &another) noexcept {
		std::swap(m_cmp, another.m_cmp);
		std::swap(m_rt, another.m_rt);
		std::swap(m_sz, another.m_sz);
	}

protected:
	using UniquePointer = std::unique_ptr<Node>;

	static ptrdiff_t s_distOf(const UniquePointer &p) noexcept {
		return (p ? p->dist : -1);
	}

	UniquePointer m_rt;
	size_t m_sz;
	Cmp m_cmp;

	UniquePointer m_merge(UniquePointer p, UniquePointer q) noexcept {
		if (!p && !q) return nullptr;
		if (p && q) {
			if (m_cmp(p->val, q->val)) std::swap(p, q);
			p->rch = m_merge(std::move(p->rch), std::move(q));
			p->rch->prt = p.get();
			if (s_distOf(p->lch) < s_distOf(p->rch)) std::swap(p->lch, p->rch);
			p->dist = s_distOf(p->rch) + 1;
		} else if (q) {
			p = std::move(q);
		}
		p->prt = nullptr;
		return p;
	}

	void m_pushUp(Node *cur) noexcept {
		for (; cur; cur = cur->prt) {
			if (s_distOf(cur->lch) < s_distOf(cur->rch)) {
				std::swap(cur->lch, cur->rch);
			}
			auto new_dist = s_distOf(cur->rch) + 1;
			if (cur->dist == new_dist) break;
			cur->dist = new_dist;
		}
	}

private:
};
```

== Fenwick Tree

```cpp
template <typename T, typename Oper = std::plus<T>,
		  typename Seq = std::vector<T>>
class FenwickTree {
public:
	/**
	 * @pre oper must be a commutative monoid operator on T.
	 *
	 * @note O(n) time complexity.
	 */
	explicit FenwickTree(Seq arr = Seq(), Oper oper = Oper())
		: m_arr(std::move(arr)), m_tree(m_arr), m_oper(std::move(oper)) {
		m_build();
	}
	void assign(Seq arr = Seq(), Oper oper = Oper()) {
		m_arr = std::move(arr);
		m_tree = m_arr;
		m_oper = std::move(oper);
		m_build();
	}

	size_t size() const noexcept { return m_tree.size(); }
	bool empty() const noexcept { return m_tree.empty(); }
	const Seq &tree() const noexcept { return m_tree; }
	const Seq &data() const noexcept { return m_arr; }

	const T &get(size_t idx) const noexcept {
		assert(idx < size());
		return m_arr[idx];
	}
	const T &operator[](size_t idx) const noexcept { return get(idx); }
	/**
	 * @brief Set the value at position `idx` to `val`.
	 * For cancellative operators, it's better to use
	 * `modify(idx, val - get(idx))`, which can be done in O(log(size())) time.
	 *
	 * @note O(log(size())^2) time complexity.
	 */
	void set(size_t idx, const T &val) {
		assert(idx < size());
		m_arr[idx] = val;
		for (++idx; idx <= size(); idx += lowbit(idx)) {
			m_tree[idx - 1] = m_arr[idx - 1];
			for (size_t i = 1; i < lowbit(idx); i <<= 1) {
				m_tree[idx - 1] = m_oper(m_tree[idx - 1], m_tree[idx - i - 1]);
			}
		}
	}
	/**
	 * @brief Modify the value at position `idx` by `diff`.
	 *
	 * @note If you want to use custom modifiers instead of the monoid operator,
	 * e.g. decrease a value in a max Fenwick tree,
	 * you can call `set(idx, modifier(get(idx), diff))` instead,
	 * which can be done in O(log(size())^2) time.
	 *
	 * @note O(log(size())) time complexity.
	 */
	void modify(size_t idx, const T &diff) {
		assert(idx < size());
		m_arr[idx] = m_oper(m_arr[idx], diff);
		for (++idx; idx <= size(); idx += lowbit(idx)) {
			m_tree[idx - 1] = m_oper(m_tree[idx - 1], diff);
		}
	}

	/**
	 * @brief Query the sum of the subarray [pos, pos + len).
	 *
	 * @param id The identity element of the monoid.
	 *
	 * @note O(log(size())) time complexity.
	 */
	T query(size_t pos, size_t len, T id = T()) const {
		assert(pos <= size());
		len = std::min(len, size() - pos);
		size_t l = pos + 1, r = pos + len;
		while (l <= r) {
			if (r - lowbit(r) + 1 >= l) {
				id = m_oper(id, m_tree[r - 1]);
				r -= lowbit(r);
			} else {
				id = m_oper(id, m_arr[r - 1]);
				--r;
			}
		}
		return id;
	}

protected:
	Seq m_arr, m_tree;
	Oper m_oper;

	void m_build() {
		for (size_t i = 1; i <= size(); ++i) {
			size_t j = i + lowbit(i);
			if (j > size()) continue;
			m_tree[j - 1] = m_oper(m_tree[j - 1], m_tree[i - 1]);
		}
	}
};
```

== Segment Tree

=== ZKW Segment Tree

```cpp
template <typename T, typename Oper = std::plus<T>>
class ZkwSegTree {
public:
	static size_t leafNumOf(size_t n) {
		return (n ? (size_t(1) << ceilLogn2(n)) : 1);
	}
	static size_t leftChildOf(size_t rt) { return (rt << 1); }
	static size_t rightChildOf(size_t rt) { return ((rt << 1) | 1); }

	/**
	 * @pre `oper` must be a monoid operator on `T` with identity element `id`.
	 */
	explicit ZkwSegTree(size_t n = 0, Oper oper = Oper(), T id = T())
		: m_sz(n), m_leaf_num(leafNumOf(n)),
		  m_tree(m_leaf_num << 1, id),
		  m_oper(std::move(oper)), m_id(std::move(id)) {}
	explicit ZkwSegTree(const std::vector<T> &arr, Oper oper = Oper(), T id = T())
		: m_sz(arr.size()), m_leaf_num(leafNumOf(arr.size())),
		  m_tree(m_leaf_num << 1, id),
		  m_oper(std::move(oper)), m_id(std::move(id)) {
		std::copy(arr.begin(), arr.end(), m_tree.begin() + m_leaf_num);
		for (size_t i = m_leaf_num - 1; i; --i) m_pushUp(i);
	}
	void assign(size_t n, Oper oper = Oper(), T id = T()) {
		m_sz = n;
		m_leaf_num = leafNumOf(n);
		m_tree.assign(m_leaf_num << 1, id);
		m_oper = std::move(oper);
		m_id = std::move(id);
	}
	void assign(const std::vector<T> &arr, Oper oper = Oper(), T id = T()) {
		m_sz = arr.size();
		m_leaf_num = leafNumOf(arr.size());
		m_tree.assign(m_leaf_num << 1, id);
		std::copy(arr.begin(), arr.end(), m_tree.begin() + m_leaf_num);
		m_oper = std::move(oper);
		m_id = std::move(id);
		for (size_t i = m_leaf_num - 1; i; --i) m_pushUp(i);
	}

	size_t size() const noexcept { return m_sz; }
	size_t leafNum() const noexcept { return m_leaf_num; }
	bool empty() const noexcept { return !m_sz; }

	const T &get(size_t pos) const {
		assert(pos < m_sz);
		return m_tree[m_leaf_num + pos];
	}
	const T &operator[](size_t pos) const { return get(pos); }
	/**
	 * @note `O(log(size()))` time complexity.
	 */
	void set(size_t pos, const T &val) {
		assert(pos < m_sz);
		pos += m_leaf_num;
		m_tree[pos] = val;
		for (pos >>= 1; pos; pos >>= 1) m_pushUp(pos);
	}

	/**
	 * @note `O(log(size()))` time complexity.
	 */
	T query(size_t pos, size_t len) const {
		assert(pos <= m_sz);
		len = std::min(len, m_sz - pos);

		if (pos == 0 && len == m_sz) return m_tree[1];

		T ls = m_id, rs = m_id;
		for (size_t l = pos + m_leaf_num, r = pos + len + m_leaf_num; l < r; l >>= 1, r >>= 1) {
			if (l & 1) ls = m_oper(ls, m_tree[l++]);
			if (r & 1) rs = m_oper(m_tree[--r], rs);
		}
		return m_oper(ls, rs);
	}

	/**
	 * @pre `pred(id) == true`.
	 *
	 * @return `beg` in `[0, end]` s.t.
	 * - `pred(query(beg, end - beg)) == true`;
	 * - `beg == 0 || pred(query(beg - 1, end - beg + 1)) == false`.
	 * If `pred` is monotone, this is the minimum `beg` in [0, end] s.t.
	 * `pred(query(beg, end - beg)) == true`.
	 *
	 * @note `O(log(size()))` time complexity.
	 */
	template <typename Pred>
	size_t minLeft(size_t end, Pred pred = Pred()) const {
		assert(end <= m_sz);
		if (!end) return 0;
		end += m_leaf_num;
		T s = m_id;
		do {
			--end;
			while (end > 1 && (end & 1)) end >>= 1;
			if (!pred(m_oper(m_tree[end], s))) {
				while (end < m_leaf_num) {
					end = rightChildOf(end);
					if (pred(m_oper(m_tree[end], s))) {
						s = m_oper(m_tree[end], s);
						--end;
					}
				}
				return (end - m_leaf_num + 1);
			}
			s = m_oper(m_tree[end], s);
		} while (lowbit(end) != end);
		return 0;
	}

	/**
	 * @pre `pred(id) == true`.
	 *
	 * @return `end` in `[beg, size()]` s.t.
	 * - `pred(query(beg, end - beg)) == true`;
	 * - `end == size() || pred(query(beg, end + 1 - beg)) == false`.
	 * If `pred` is monotone, this is the maximum `end` in [beg, size()] s.t.
	 * `pred(query(beg, end - beg)) == true`.
	 *
	 * @note `O(log(size()))` time complexity.
	 */
	template <typename Pred>
	size_t maxRight(size_t beg, Pred pred = Pred()) const {
		assert(beg <= m_sz);
		if (beg == m_sz) return m_sz;
		beg += m_leaf_num;
		T s = m_id;
		do {
			while (((beg & 1) == 0)) beg >>= 1;
			if (!pred(m_oper(s, m_tree[beg]))) {
				while (beg < m_leaf_num) {
					beg = leftChildOf(beg);
					if (pred(m_oper(s, m_tree[beg]))) {
						s = m_oper(s, m_tree[beg]);
						++beg;
					}
				}
				return std::min(m_sz, beg - m_leaf_num);
			}
			s = m_oper(s, m_tree[beg]);
			++beg;
		} while (lowbit(beg) != beg);
		return m_sz;
	}

protected:
	size_t m_sz, m_leaf_num;
	std::vector<T> m_tree;
	Oper m_oper;
	T m_id;

	void m_pushUp(size_t rt) {
		m_tree[rt] = m_oper(m_tree[leftChildOf(rt)], m_tree[rightChildOf(rt)]);
	}
};
```

=== ZKW Lazy Segment Tree

```cpp
template <typename T, typename LazyTag = T, typename Oper = std::plus<T>, typename ApplyTag = std::plus<T>, typename MergeTags = std::plus<LazyTag>>
class ZkwLazySegTree {
public:
	static size_t leafNumOf(size_t n) {
		return (n ? (size_t(1) << ceilLogn2(n)) : 1);
	}
	static size_t leftChildOf(size_t rt) { return (rt << 1); }
	static size_t rightChildOf(size_t rt) { return ((rt << 1) | 1); }

	/**
	 * @pre `oper` must be a monoid operator on `T` with identity element `id`.
	 *
	 * @param `apply_tag` applies a `LazyTag` to a node `T` (passed as
	 * `(tag, val)`) and returns the updated node value.
	 * @param `merge_tags` combines two `LazyTag` elements (passed as
	 * `(new_tag, old_tag)`).
	 *
	 * @warning If `ApplyTag` depends on range length, which should be stored
	 * in `T`, construct the tree from `std::vector<T>`, or all nodes will
	 * permanently have `len == 0`.
	 *
	 * @note `O(size())` time complexity.
	 */
	explicit ZkwLazySegTree(size_t n = 0, Oper oper = Oper(), T id = T(), ApplyTag apply_tag = ApplyTag(), MergeTags merge_tags = MergeTags(), LazyTag lazy_id = LazyTag())
		: m_sz(n),
		  m_leaf_num(leafNumOf(n)), m_height(n ? ceilLogn2(m_leaf_num) : 0),
		  m_tree(m_leaf_num << 1, id), m_tags(m_leaf_num, lazy_id),
		  m_oper(std::move(oper)), m_id(std::move(id)),
		  m_apply_tag(std::move(apply_tag)),
		  m_merge_tags(std::move(merge_tags)),
		  m_lazy_id(std::move(lazy_id)) {}
	explicit ZkwLazySegTree(const std::vector<T> &arr, Oper oper = Oper(), T id = T(), ApplyTag apply_tag = ApplyTag(), MergeTags merge_tags = MergeTags(), LazyTag lazy_id = LazyTag())
		: m_sz(arr.size()),
		  m_leaf_num(leafNumOf(arr.size())),
		  m_height(arr.empty() ? 0 : ceilLogn2(m_leaf_num)),
		  m_tree(m_leaf_num << 1, id),
		  m_tags(m_leaf_num, lazy_id),
		  m_oper(std::move(oper)),
		  m_id(std::move(id)),
		  m_apply_tag(std::move(apply_tag)),
		  m_merge_tags(std::move(merge_tags)),
		  m_lazy_id(std::move(lazy_id)) {
		std::copy(arr.begin(), arr.end(), m_tree.begin() + m_leaf_num);
		for (size_t i = m_leaf_num - 1; i; --i) m_pushUp(i);
	}
	void assign(size_t n, Oper oper = Oper(), T id = T(), ApplyTag apply_tag = ApplyTag(), MergeTags merge_tags = MergeTags(), LazyTag lazy_id = LazyTag()) {
		m_sz = n;
		m_leaf_num = leafNumOf(n);
		m_height = (n ? ceilLogn2(m_leaf_num) : 0);
		m_tree.assign(m_leaf_num << 1, id);
		m_tags.assign(m_leaf_num, lazy_id);
		m_oper = std::move(oper);
		m_id = std::move(id);
		m_apply_tag = std::move(apply_tag);
		m_merge_tags = std::move(merge_tags);
		m_lazy_id = std::move(lazy_id);
	}
	void assign(const std::vector<T> &arr, Oper oper = Oper(), T id = T(), ApplyTag apply_tag = ApplyTag(), MergeTags merge_tags = MergeTags(), LazyTag lazy_id = LazyTag()) {
		m_sz = arr.size();
		m_leaf_num = leafNumOf(arr.size());
		m_height = (arr.empty() ? 0 : ceilLogn2(m_leaf_num));
		m_tree.assign(m_leaf_num << 1, id);
		std::copy(arr.begin(), arr.end(), m_tree.begin() + m_leaf_num);
		m_tags.assign(m_leaf_num, lazy_id);
		m_oper = std::move(oper);
		m_id = std::move(id);
		m_apply_tag = std::move(apply_tag);
		m_merge_tags = std::move(merge_tags);
		m_lazy_id = std::move(lazy_id);
		for (size_t i = m_leaf_num - 1; i; --i) m_pushUp(i);
	}

	size_t size() const noexcept { return m_sz; }
	size_t leafNum() const noexcept { return m_leaf_num; }
	/**
	 * @return number of edges from the root to the leaves.
	 */
	size_t height() const noexcept { return m_height; }
	bool empty() const noexcept { return !m_sz; }

	/**
	 * @note Not `const` because `m_pushDown` modifies internal states.
	 * @note `O(log(size()))` time complexity.
	 */
	const T &get(size_t pos) {
		assert(pos < m_sz);
		pos += m_leaf_num;
		for (size_t i = m_height; i; --i) m_pushDown(pos >> i);
		return m_tree[pos];
	}
	const T &operator[](size_t pos) { return get(pos); }

	/**
	 * @note `O(log(size()))` time complexity.
	 */
	void set(size_t pos, const T &val) {
		assert(pos < m_sz);
		pos += m_leaf_num;
		for (size_t i = m_height; i; --i) m_pushDown(pos >> i);
		m_tree[pos] = val;
		for (size_t i = 1; i <= m_height; ++i) m_pushUp(pos >> i);
	}

	/**
	 * @note `O(log(size()))` time complexity.
	 */
	T query(size_t pos, size_t len) {
		assert(pos <= m_sz);
		len = std::min(len, m_sz - pos);

		if (len == 0) return m_id;
		if (pos == 0 && len == m_sz) return m_tree[1];

		size_t l = pos + m_leaf_num;
		size_t r = pos + len + m_leaf_num;

		for (size_t i = m_height; i; --i) {
			if (((l >> i) << i) != l) m_pushDown(l >> i);
			if (((r >> i) << i) != r) m_pushDown((r - 1) >> i);
		}

		T ls = m_id, rs = m_id;
		for (; l < r; l >>= 1, r >>= 1) {
			if (l & 1) ls = m_oper(ls, m_tree[l++]);
			if (r & 1) rs = m_oper(m_tree[--r], rs);
		}
		return m_oper(ls, rs);
	}

	/**
	 * @note Point application. `O(log(size()))` time complexity.
	 */
	void apply(size_t pos, const LazyTag &tag) {
		assert(pos < m_sz);
		pos += m_leaf_num;
		for (size_t i = m_height; i; --i) m_pushDown(pos >> i);
		m_tree[pos] = m_apply_tag(tag, m_tree[pos]);
		for (size_t i = 1; i <= m_height; ++i) m_pushUp(pos >> i);
	}

	/**
	 * @note Range application. `O(log(size()))` time complexity.
	 */
	void apply(size_t pos, size_t len, const LazyTag &tag) {
		assert(pos <= m_sz);
		len = std::min(len, m_sz - pos);
		if (len == 0) return;

		size_t l = pos + m_leaf_num;
		size_t r = pos + len + m_leaf_num;

		for (size_t i = m_height; i; --i) {
			if (((l >> i) << i) != l) m_pushDown(l >> i);
			if (((r >> i) << i) != r) m_pushDown((r - 1) >> i);
		}

		{
			size_t l2 = l, r2 = r;
			for (; l < r; l >>= 1, r >>= 1) {
				if (l & 1) m_apply(l++, tag);
				if (r & 1) m_apply(--r, tag);
			}
			l = l2;
			r = r2;
		}

		for (size_t i = 1; i <= m_height; ++i) {
			if (((l >> i) << i) != l) m_pushUp(l >> i);
			if (((r >> i) << i) != r) m_pushUp((r - 1) >> i);
		}
	}

	/**
	 * @pre `pred(id) == true`.
	 *
	 * @return Minimum `beg` in `[0, end]` s.t. `pred(query(beg, end - beg)) == true`.
	 *
	 * @note `O(log(size()))` time complexity.
	 */
	template <typename Pred>
	size_t minLeft(size_t end, Pred pred = Pred()) {
		assert(end <= m_sz);
		assert(pred(m_id));
		if (!end) return 0;

		end += m_leaf_num;
		for (size_t i = m_height; i; --i) m_pushDown((end - 1) >> i);

		T s = m_id;
		do {
			--end;
			while (end > 1 && (end & 1)) end >>= 1;
			if (!pred(m_oper(m_tree[end], s))) {
				while (end < m_leaf_num) {
					m_pushDown(end);
					end = rightChildOf(end);
					if (pred(m_oper(m_tree[end], s))) {
						s = m_oper(m_tree[end], s);
						--end;
					}
				}
				return end - m_leaf_num + 1;
			}
			s = m_oper(m_tree[end], s);
		} while (lowbit(end) != end);
		return 0;
	}

	/**
	 * @pre `pred(id) == true`.
	 *
	 * @return Maximum `end` in `[beg, size()]` s.t. `pred(query(beg, end - beg)) == true`.
	 *
	 * @note `O(log(size()))` time complexity.
	 */
	template <typename Pred>
	size_t maxRight(size_t beg, Pred pred = Pred()) {
		assert(beg <= m_sz);
		assert(pred(m_id));
		if (beg == m_sz) return m_sz;

		beg += m_leaf_num;
		for (size_t i = m_height; i; --i) m_pushDown(beg >> i);

		T s = m_id;
		do {
			while ((beg & 1) == 0) beg >>= 1;
			if (!pred(m_oper(s, m_tree[beg]))) {
				while (beg < m_leaf_num) {
					m_pushDown(beg);
					beg = leftChildOf(beg);
					if (pred(m_oper(s, m_tree[beg]))) {
						s = m_oper(s, m_tree[beg]);
						++beg;
					}
				}
				return std::min(m_sz, beg - m_leaf_num);
			}
			s = m_oper(s, m_tree[beg]);
			++beg;
		} while (lowbit(beg) != beg);
		return m_sz;
	}

protected:
	size_t m_sz, m_leaf_num, m_height;
	std::vector<T> m_tree;
	std::vector<LazyTag> m_tags;
	Oper m_oper;
	T m_id;
	ApplyTag m_apply_tag;
	MergeTags m_merge_tags;
	LazyTag m_lazy_id;

	void m_pushUp(size_t rt) {
		m_tree[rt] = m_oper(m_tree[leftChildOf(rt)], m_tree[rightChildOf(rt)]);
	}
	void m_apply(size_t rt, const LazyTag &tag) {
		m_tree[rt] = m_apply_tag(tag, m_tree[rt]);
		if (rt < m_leaf_num) m_tags[rt] = m_merge_tags(tag, m_tags[rt]);
	}
	void m_pushDown(size_t rt) {
		m_apply(leftChildOf(rt), m_tags[rt]);
		m_apply(rightChildOf(rt), m_tags[rt]);
		m_tags[rt] = m_lazy_id;
	}
};
```

== GNU Policy-Based Data Structures

PBDS does not support container interfaces introduced in C++11 and later standards. Specifically, PBDS containers do not support
- Construction from `std::initialization_list` or a right-value reference.
- `emplace`, `emplace_hint`, `insert(value_type &&)`, etc. that rely on move semantics.
- Interfaces that rely on ranges,  like `insert_range`.

=== Invalidation Guarantees

There are two levels of iterator capabilities in PBDS:
+ A #emph[point-type] iterator is an iterator that refers to a specific element, as one returned through an associative-container's `find` method. Point-type iterators does not support any movement like `++`.
+ A #emph[range-type] iterator is an iterator that is used to go over a sequence of elements, as returned by a container's `begin` method.

These are conceptual names and correspond to types `point_iterator` and `iterator` respectively.

#figure(
  caption: [Invalidation Guarantees of PBDS],
)[
  #table(
    align: (center, left),
    columns: 2,
    strong[Tag], strong[Guarantee],
    `basic_invalidation_guarantee`,
    [Guarantees validation only when the container is not modified. Any modification to the container structure may cause invalidation.],

    `point_invalidation_guarantee`,
    [On the basis of `basic_invalidation_guarantee`, existing iterators can still be used for dereference as long as their corresponding elements are not removed. However, existing range-type iterators should #strong[not] be used for iteration anymore.],

    `range_invalidation_guarantee`,
    [On the basis of `point_invalidation_guarantee`, existing range-type iterators are guaranteed to remain in correct relative positions and can continue to be used for iteration.],
  )
]<tab:pbds-invalidation-guarantees>

=== Associative Containers

Use `__gnu_pbds::null_type` as the mapped type in an associative container to create sets.

Associative containers do #strong[not] support multiset or multimap. Use `std::pair` as keys when needed.

==== Balanced Binary Trees

```cpp
__gnu_pbds::tree<Key, Val, Cmp,
	TreeTag, __gnu_pbds::tree_order_statistics_node_update> tree;
```

On the basis of `std::map`, `tree` in PBDS additionally support
- `find_by_order` (0-based) and `order_of_key` (querying the number of keys strictly less than the given key) when `__gnu_pbds::tree_order_statistics_node_update` is specified.
- Splitting and joining of trees.

There are 3 kinds of implementation, specified by the tag:
#figure(
  caption: [Three Tags of `__gnu_pbds::tree`],
)[
  #table(
    align: (center, center, left),
    columns: 3,
    strong[Tag], strong[Structure], strong[Feature],
    `rb_tree_tag`, [Red-Black Tree], [All operations can be done in $O(log n)$ time in worst cases.],
    `splay_tree_tag`, [Splay Tree], [The average time complexity of each operation is $O(log n)$.],
    `ov_tree_tag`, [Ordered Vector], [$O(log n)$ finding, and $O(n)$ insertion and removal.],
  )
]<tab:pbds-tree-tags>

==== Hash Tables

#figure(
  caption: [Hash Tables in PBDS],
)[
  #table(
    align: (center, center, left),
    columns: 3,
    strong[Container], strong[Collision Handling], strong[Feature],
    `gp_hash_table`,
    [Open Addressing],
    [`gp` stands for #strong[g]eneral-#strong[p]robing.\ Faster.\ Only provides `basic_invalidation_guarantee`.],

    `cc_hash_table`,
    [#strong[C]ollision #strong[C]haining],
    [Provides stronger invalidation guarantee (`point_invalidation_guarantee`).],
  )
]<tab:pbds-hash-tables>

Note that although `size` returns the number of elements stored in a hash table, `resize` operates on the underlying structure, setting the number of its #strong[slots] (for `gp_hash_table`) or #strong[buckets] (for `cc_hash_table`) to the given `new_size`. It is different from `std::unordered_map<>::reserve`, which reserves the space for the given number of #strong[elements]\; `resize` does #strong[not] handle the load factor automatically. That is, $ceil(n / lambda)$ slots/buckets are required to reserve space for $n$ #strong[elements], where $lambda$ is the #strong[load factor].

==== Trie

#figure(
  caption: [Template Parameters of `__gnu_pbds::trie`],
)[
  #table(
    columns: 4,
    align: (center, center, left, left),

    strong[Parameter], strong[Meaning], strong[Requirements], strong[Examples],
    `Key`, [Key Type], [Copy-constructible.], [`std::string`],
    `Mapped`,
    [Value Type],
    [Copy-constructible. `operator[]` requires default-constructibility in case the queried key does not exist.],
    [`__gnu_pbds::null_type`, `int`, etc.],

    `_ATraits`,
    [Element Access Traits],
    [
      - Types:
        - `size_type`.
        - `key_type`.
        - `const_iterator`: a #strong[forward iterator] to iterate in `key_type`.
      - Values:
        - `enum { max_size }`: the #strong[compile-time] domain size of elements in keys.
      - Functions:
        - `static const_iterator begin(const key_type &)`.
      - `static const_iterator end(const key_type &)`.
      - `size_type e_pos(*const_iterator)`: static or non-static const member function.
    ],
    table.cell(breakable: true)[
      ```cpp
       __gnu_pbds::
       	trie_string_access_traits<
       		std::vector<int>, 0, 9
      >
      ```
    ],

    `Tag`, [Tag of Trie Type], [Only supports `pat_trie_tag`.], [],
    `Node_Update`,
    [Node Update Policy],
    [Only supports `null_node_update`, `trie_prefix_search_node_update` and `trie_order_statistics_node_update`],
    [],
    `_Alloc`, [Allocator], [Omitted.], [`std::allocator`],
  )
]<tab:trie-template-parameters>

There are three node update policy:
- `null_node_update`: provides only the basic interfaces.
- `trie_prefix_search_node_update`: provides `prefix_range`, which returns a pair of iterators marking the left-closed right-open interval.
- `trie_order_statistics_node_update`: provides `find_by_order` (0-based) and `order_of_key` (number of keys strictly less than the given key).
#h(-indent) Different policies #strong[cannot] be stacked. A solution is to maintain a container of each additional policy.

==== Self-Organizing Linked-List

There are two update policies supported:
- `lu_move_to_front_policy`: unconditionally moves an element to the front of the list every time it is accessed.
- `lu_counter_policy`: Each element keeps a counter; when the counter reaches a threshold (`Max_Count`, default `5`), the element is moved to the front.

These policies does not change the asymptotic complexity of linked-list operations. This container is suitable only for numerous small-scale secondary maps, e.g. as the mapped type in a hash table.

Note that `list_update::size` is of $Theta(n)$ time complexity.

=== Heap

On the basis of `std::priority_queue`, `__gnu_pbds` additionally supports
-
- `push` returns a #strong[point-type] iterator of the inserted element.
- `erase`: remove the element specified by a given iterator.
- `join`: join two heaps.
- `modify`: modify the element specified by a given iterator.
- `begin` and `end`: return #strong[range-type] iterators.

There are five types of heap implementation, specified by the tag:
#figure(
  caption: [Heap Tags of PBDS],
)[
  #show raw: it => text(
    it.text.replace("_", "_\u{200b}"),
  )

  #table(
    columns: 8,
    align: center + horizon,

    strong[Tag], strong[Underlying Structure], strong[Invalidation Guarantee], `push`, `pop`, `modify`, `erase`, `join`,
    `binary_heap_tag`,

    [Array-Based Binary Heap],
    [Basic],
    table.cell(colspan: 2)[Amortized $Theta(log n)$. Worst $Theta(n)$ (from capacity growth).],
    table.cell(colspan: 3)[$Theta(n)$],

    `pairing_heap_tag`,
    [Paring Heap],
    table.cell(rowspan: 4)[Point],
    $O(1)$,
    table.cell(colspan: 3)[Amortized $Theta(log n)$. Worst $Theta(n)$.],
    $O(1)$,

    `binomial_heap_tag`,
    [Binomial Heap],
    [Amortized $O(1)$. Worst $Theta(log n)$.],
    table.cell(colspan: 4)[$Theta(log n)$],

    `rc_binomial_heap_tag`, [Redundant Counter Binomial Heap], $O(1)$, table.cell(colspan: 4)[$Theta(log n)$],

    `thin_heap_tag`,
    [Thin Heap],
    $O(1)$,
    [Amortized $Theta(log n)$. Worst $Theta(n).$],
    [
      - Increase: Amortized $O(1)$. Worst $Theta(log n)$.
      - Decrease: $O(log n)$.
    ],
    [Amortized $Theta(log n)$. Worst $Theta(n)$.],
    $Theta(n)$,
  )
]<tab:pbds-heap-tags>

= Number Theory

== Cantor Expansion

```cpp
/**
 * @pre arr.size() <= 20.
 *
 * @return the rank (0-indexed) of arr in all its permutation.
 *
 * @note O(n log n) time complexity.
 */
template <typename T, typename Cmp = std::less<T>, typename Eq = std::equal_to<T>>
uint64_t cantorExpand(const std::vector<T> &arr, Cmp cmp = Cmp(), Eq eq = Eq()) {
	std::vector<size_t> rks;
	size_t n = arr.size(),
		   tot = discretize(arr.begin(), arr.end(), rks, cmp, eq).size();
	FenwickTree<size_t> tree{std::vector<size_t>(tot)};
	std::vector<size_t> cnts(tot);
	uint64_t res = 0;
	__uint128_t perm_num = 1; // number of distinct permutations of the suffix
	for (size_t i = n; i--;) {
		++cnts[rks[i]];
		res += perm_num * tree.query(0, rks[i]) / cnts[rks[i]];
		perm_num = perm_num * (n - i) / cnts[rks[i]];
		tree.modify(rks[i], 1);
	}
	return res;
}
```

== Quick Power

```cpp
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
```

== Extended Euclidean Algorithm

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

== Modular Multiplicative Inverse

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
 * @return modular multiplicative inverse of [0, max].
 * @pre assuming mod is a prime; inverse of multiples of mod (including 0) is
 * 	undefined (0 as the placeholder).
 */
std::vector<uint64_t> modMulInvs(uint32_t max, uint64_t mod) {
	std::vector<uint64_t> invs(max + 1);
	if (!max) return invs;
	invs[1] = 1;
	uint64_t i = 2;
	for (; i < mod && i <= max; ++i) {
		invs[i] = static_cast<__uint128_t>(mod - mod / i) * invs[mod % i] % mod;
	}
	for (; i <= max; ++i) {
		invs[i] = invs[i % mod];
	}
	return invs;
}

/**
 * @return modular multiplicative inverse of each element of arr.
 * @pre assuming mod is a prime and none of arr is divisible by mod.
 */
std::vector<uint64_t> modMulInvs(const std::vector<uint64_t> &arr, uint64_t mod) {
	if (arr.empty()) return arr;
	auto pref_prods = std::vector<uint64_t>(arr.size() + 1);
	pref_prods[0] = 1;
	for (size_t i = 0; i != arr.size(); ++i) {
		pref_prods[i + 1] = static_cast<__uint128_t>(pref_prods[i]) * arr[i] % mod;
	}
	__uint128_t prod_inv = modMulInv(pref_prods.back(), mod);
	auto invs = std::vector<uint64_t>(arr.size());
	for (size_t i = arr.size() - 1; ~i; --i) {
		invs[i] = pref_prods[i] * prod_inv % mod;
		prod_inv = prod_inv * arr[i] % mod;
	}
	return invs;
}
```

== Extended Chinese Remainder Theorem

Solve the system of congruences
$
  x equiv r_i space (mod m_i) space (i in NN inter [0,n)).
$
#h(-indent) $op(lcm)_(i = 0)^(n - 1) m_i < 10^18$ should be guaranteed to avoid overflow.

This is an #strong[online] algorithm, i.e. the equations are be added one by one.

```cpp
/**
 * @brief solve the system of equations x % mod == rem for each (mod, rem), the
 * solution expressed as x = period * n + ans for any integer n. This is an online
 * algorithm. Each call to this function adds a new equation to the system. You
 * should initialize period = 1, ans = 0 before any equation is added.
 * @return true if a solution exists, false otherwise
 */
inline bool exCrt(
	int64_t mod, int64_t rem,
	int64_t &period, int64_t &ans
) {
	if (mod == 0) return false;
	if (mod < 0) mod = -mod;
	rem = (rem % mod + mod) % mod;

	int64_t x, y, g = exGcd(period, mod, x, y);
	if ((rem - ans) % g != 0) return false;
	auto q = mod / g;
	x = (((rem - ans) / g * static_cast<__int128_t>(x)) % q + q) % q;

	auto new_period = period * (mod / g);
	ans = (ans + period * x) % new_period;
	period = new_period;
	return true;
}
```

- Time complexity: $O(log m_i)$ for a single call. $O(n log(max_(i = 0)^(n - 1) m_i))$ for the entire equation system.
- Space complexity: $O(1)$.

== Derangement

#definition[Derangement][
  A derangement is a permutation of the elements of a set, such that no element appears in its original position.
]<def:derangement>

The number of derangements of $n in NN$ distinct elements, denoted by $D_n$, satisfies
$
  D_n = n! sum_(i = 0)^n (-1)^i / i! = floor(n! / "e" + 1/2) \
  D_n = n D_(n - 1) + (-1)^n space (n >= 1) \
  D_n = (n - 1)(D_(n - 1) + D_(n - 2)) space (n >= 2)
$

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

== Prime Sieves

```cpp
std::vector<uint32_t> sieveOfEratosthenes(uint32_t max) {
	if (max <= 1) return {};
	auto not_prime = std::vector<bool>(max + 1);
	std::vector<uint32_t> primes;
	primes.reserve(max ? (3 * max / (floorLogn2(max) << 1)) : 0); // approximate number of primes
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
	primes.reserve(max ? (3 * max / (floorLogn2(max) << 1)) : 0); // approximate number of primes
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
```

#definition[Multiplicative Function][
  A function $f$ on $NN^*$ is called #emph[multiplicative] if
  - $f(1) = 1$.
  - $(forall x,y in NN^* "s.t." x perp y) space (f(x dot y) = f(x) dot f(y))$.
]<def:multiplicative-function>

#definition[Completely Multiplicative Function][
  A function $f$ on $NN^*$ is called #emph[completely multiplicative] if
  - $f(1) = 1$.
  - $(forall x,y in NN^*) space (f(x dot y) = f(x) dot f(y))$.
]<def:completely-multiplicative-function>

The linear sieve can be extended to generate values of any #strong[multiplicative] function $f$. If $f(p^e)$ can be computed in $T(p, e)$ time, then the total time complexity to compute $f[1, n]$ is approximately
$
  O(n)+O(n / (ln n) T(n,1)).
$

```cpp
/**
 * @brief linear sieve to generate primes and multiplicative function values
 * @return tuple of (not_prime, primes, low, func_values)
 * 	not_prime: bool array indicating whether i is not prime
 * 	primes: list of all primes <= max
 * 	low: low[i] = the largest power of the minimal prime factor of i that divides i
 * 	func_values: func_values[i] = func applied on the prime factorization of i;
 * 		func_values[0] is is value-initialized and does not represent f(0).
 * @tparam Ret return type of func
 * @tparam Func function to generate multiplicative function values
 * @param max maximum value to sieve
 * @param func function with signature Ret(uint32_t prime, uint32_t prime_pow)
 * 	that func(p, p^e) is defined as f(p^e), where f is the multiplicative
 * 	function, p is a prime number and e is an non-negative integer. Particularly,
 * 	func(1, 1) should return f(1), which must satisfy f(n) == f(1) * f(n) for
 * 	every positive integer n. Under the usual definition of multiplicative
 * 	functions, f(1) = 1.
 */
template <typename Ret, typename Func>
inline std::tuple<std::vector<bool>, std::vector<uint32_t>, std::vector<uint32_t>, std::vector<Ret>>
linearSieve(uint32_t max, Func func = Func()) {
	if (max == 0) return {{true}, {}, {0}, {Ret()}};

	std::vector<bool> not_prime(max + 1);
	std::vector<uint32_t> primes;
	primes.reserve((max >= 2) ? (3 * max / (floorLogn2(max) << 1)) : 0); // approximate number of primes
	std::vector<uint32_t> low(max + 1);
	std::vector<Ret> res(max + 1);

	not_prime[0] = not_prime[1] = true;
	low[1] = 1;
	res[1] = func(1, 1);
	for (uint32_t i = 2; i <= max; ++i) {
		if (!not_prime[i]) {
			primes.emplace_back(i);
			low[i] = i;
			res[i] = func(i, i);
		}
		for (auto prime : primes) {
			auto x = i * prime;
			if (x > max) break;
			not_prime[x] = true;
			if (i % prime) {
				low[x] = prime;
				res[x] = res[prime] * res[i];
			} else { // the minimal prime factor of x is prime
				low[x] = low[i] * prime;
				if (x == low[x]) {
					res[x] = func(prime, x);
				} else {
					res[x] = res[low[x]] * res[x / low[x]];
				}
				break;
			}
		}
	}
	return {std::move(not_prime),
			std::move(primes),
			std::move(low),
			std::move(res)};
}
```

== Euler's Totient Function

Euler's totient function $phi(n)$ of a positive integer $n$ is defined as
$
  phi(n) = sum_(i = 1)^n (gcd(i, n) = 1),
$
#h(-indent) i.e. the number of positive integers that are less than or equal to $n$ and coprime to $n$.

Written in the form of Dirichlet convolution,
Euler's totient function
$
  phi = "id" * mu,
$
#h(-indent) where $id: n |-> n$ and $mu$ is the Möbius function.

```cpp
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
```

== Primality Test and Factorization

```cpp
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
```

= Polynomial

== Fast Fourier Transform

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
```

=== Usage

```cpp
inline void solve() {
	size_t m, n;
	std::cin >> m >> n;
	std::vector<Complex> a(++m), b(++n);
	for (auto &x : a) std::cin >> x;
	for (auto &x : b) std::cin >> x;
	a.resize(1 << ceilLogn2(m + n - 1));
	b.resize(a.size());
	fft(a);
	fft(b);
	for (size_t i = 0; i != a.size(); ++i) a[i] *= b[i];
	fft(a, true);
	a.resize(n + m - 1);
	for (auto &x : a) std::cout << std::lround(x.real()) << ' ';
}
```

= Sequence

== Merge Sort Counting Inversions

```cpp
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
```

== Discretization

```cpp
/**
 * @return sorted unique elements.
 *
 * @note O(n log n) time complexity.
 */
template <typename Iter,
		  typename Cmp = std::less<typename std::iterator_traits<Iter>::value_type>,
		  typename Eq = std::equal_to<typename std::iterator_traits<Iter>::value_type>,
		  typename = RequireFwdIter<Iter>>
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

== Longest Increasing Subsequence

```cpp
/**
 * @return indices of elements consisting one longest increasing subsequence of
 * [@param begin, @param end).
 *
 * @note O(n log n) time complexity.
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
	if (pre.empty()) return res;
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
```

== Mo's Algorithm

```cpp
namespace mo {
	namespace std {
		struct Query {
			size_t beg, end; // `[beg, end)`
		};

		/**
		 * @pre `State` must implement
		 * - `add(size_t pos);`
		 * - `sub(size_t pos);`
		 * - `Res get();`.
		 *
		 * @note `O(n * sqrt(q) * t + q log q)` time complexity, where
		 * `q = qrys.size()` and `t` is the time complexity of state transisions.
		 */
		template <typename Res, typename State>
		::std::vector<Res> solve(size_t n,
								 const ::std::vector<Query> &qrys,
								 State &state) {
			size_t q = qrys.size();
			if (q == 0) return {};

			size_t b = ::std::max<size_t>(1, n / ::std::sqrt(q));
			::std::vector<size_t> ord(q);
			for (size_t i = 0; i < q; ++i) ord[i] = i;
			::std::sort(ord.begin(), ord.end(),
						[&](size_t lhs, size_t rhs) {
							size_t lb = qrys[lhs].beg / b, rb = qrys[rhs].beg / b;
							if (lb != rb) return lb < rb;
							return ((lb & 1)
										? (qrys[lhs].end < qrys[rhs].end)
										: (qrys[lhs].end > qrys[rhs].end));
						});

			::std::vector<Res> res(q);
			size_t beg = 0, end = 0;
			for (size_t id : ord) {
				const auto &qry = qrys[id];
				assert(qry.beg <= qry.end && qry.end <= n);
				while (beg > qry.beg) state.add(--beg);
				while (end < qry.end) state.add(end++);
				while (beg < qry.beg) state.sub(beg++);
				while (end > qry.end) state.sub(--end);
				res[id] = state.get();
			}
			return res;
		}
	}

	namespace upd {
		struct Query {
			size_t beg, end; // `[beg, end)`
			size_t tm;		 // `tm` updates need to be performed before this query
		};

		/**
		 * @pre `State` must implement
		 * - `add(size_t pos);`
		 * - `sub(size_t pos);`
		 * - `apply(size_t tm, size_t beg, size_t end);`
		 * - `undo(size_t tm, size_t beg, size_t end);`
		 * - `Res get();`
		 *
		 * @param u number of updates.
		 *
		 * @note `O(cbrt(n^2 * q^2 * u) * t + q log q)` time complexity, where
		 * `q = qrys.size()` and `t` is the time complexity of state transisions.
		 */
		template <typename Res, typename State>
		::std::vector<Res> solve(size_t n,
								 size_t u,
								 const ::std::vector<Query> &qrys,
								 State &state) {
			size_t q = qrys.size();
			if (q == 0) return {};

			size_t b = ::std::max<size_t>(
				1,
				(u
					 ? ::std::round(::std::cbrt(
						   static_cast<long double>(n) * n * u / q))
					 : n / ::std::sqrt(q)));
			::std::vector<size_t> ord(q);
			for (size_t i = 0; i < q; ++i) ord[i] = i;
			::std::sort(ord.begin(), ord.end(),
						[&](size_t lhs, size_t rhs) {
							size_t lbb = qrys[lhs].beg / b, rbb = qrys[rhs].beg / b;
							if (lbb != rbb) return lbb < rbb;
							size_t leb = qrys[lhs].end / b, reb = qrys[rhs].end / b;
							if (leb != reb) return (lbb & 1) ? leb < reb : leb > reb;
							return ((leb & 1)
										? (qrys[lhs].tm < qrys[rhs].tm)
										: (qrys[lhs].tm > qrys[rhs].tm));
						});

			::std::vector<Res> res(q);
			size_t beg = 0, end = 0, tm = 0;
			for (size_t id : ord) {
				const auto &qry = qrys[id];
				assert(qry.beg <= qry.end && qry.end <= n);
				while (beg > qry.beg) state.add(--beg);
				while (end < qry.end) state.add(end++);
				while (beg < qry.beg) state.sub(beg++);
				while (end > qry.end) state.sub(--end);
				while (tm < qry.tm) state.apply(tm++, beg, end);
				while (tm > qry.tm) state.undo(--tm, beg, end);
				res[id] = state.get();
			}
			return res;
		}
	}

	namespace rb {
		struct Query {
			size_t beg, end; // [beg, end)
		};

		/**
		 * Suitable for maintaining information that is "easy to add but hard
		 * to subtract" or "easy to remove but hard to add" (using add-only as
		 * an example).
		 *
		 * @pre `State` must implement
		 * - `add(size_t pos);`
		 * - `Res get();`
		 * - `void snapshot();` records a snapshot of the current internal state.
		 * - `void rollback();` rolls back the state to the most recent snapshot.
		 * - `void reset();`  resets the state to the initial empty/zero state.
		 *
		 * @note `O(n * sqrt(q) * t_add + q * t_snap + q log q)` time complexity,
		 * where `q = qrys.size()` and `t_add` and `t_snap` are the time
		 * complexities of `add` and `snapshot` operations, respectively.
		 */
		template <typename Res, typename State>
		::std::vector<Res> solve(size_t n,
								 const ::std::vector<Query> &qrys,
								 State &state) {
			size_t q = qrys.size();
			if (q == 0) return {};

			size_t b = ::std::max<size_t>(1, n / ::std::sqrt(q));

			::std::vector<Res> res(q);
			::std::vector<size_t> ord;
			ord.reserve(q);

			// Pre-process short queries
			for (size_t i = 0; i < q; ++i) {
				const auto &qry = qrys[i];
				assert(qry.beg <= qry.end && qry.end <= n);
				if (qry.beg == qry.end) {
					res[i] = state.get();
					continue;
				}

				// Short query
				if (qry.end - qry.beg <= b || qry.beg / b == (qry.end - 1) / b) {
					state.snapshot();
					for (size_t j = qry.beg; j < qry.end; ++j) state.add(j);
					res[i] = state.get();
					state.rollback(); // Restores state back to completely empty
				} else {
					// Long cross-block query: queue for Mo's processing
					ord.emplace_back(i);
				}
			}

			// Sort only the remaining long queries
			::std::sort(ord.begin(), ord.end(),
						[&](size_t lhs, size_t rhs) {
							size_t lb = qrys[lhs].beg / b, rb = qrys[rhs].beg / b;
							if (lb != rb) return lb < rb;
							return (qrys[lhs].end < qrys[rhs].end);
						});

			size_t last_block = -1;
			size_t beg = 0, end = 0;
			for (size_t id : ord) {
				const auto &qry = qrys[id];
				size_t block = qry.beg / b;

				// Cross-block query
				if (block != last_block) {
					state.reset();
					beg = (block + 1) * b;
					end = beg;
					last_block = block;
				}

				while (end < qry.end) state.add(end++);

				state.snapshot();
				for (size_t i = beg; i > qry.beg; state.add(--i));

				res[id] = state.get();
				state.rollback();
			}
			return res;
		}
	}
}
```

== Prefix Function

```cpp
template <typename Iter, typename = RequireFwdIter<Iter>>
inline std::vector<size_t> prefFuncOf(Iter begin, Iter end) {
	auto pi = std::vector<size_t>(std::distance(begin, end));
	if (pi.empty()) return pi;
	end = std::next(begin);
	for (size_t i = 1, j; i < pi.size(); ++i, ++end) {
		for (j = pi[i - 1]; j && *end != *std::next(begin, j); j = pi[j - 1]);
		pi[i] = j + (*end == *std::next(begin, j));
	}
	return pi;
}

template <typename TxtIter, typename PtnIter,
		  typename = RequireFwdIter<TxtIter>, typename = RequireFwdIter<PtnIter>>
inline std::vector<size_t>
kmp(TxtIter txt_beg, TxtIter txt_end,
	PtnIter ptn_beg, PtnIter ptn_end) {
	size_t txt_sz = std::distance(txt_beg, txt_end),
		   ptn_sz = std::distance(ptn_beg, ptn_end);
	std::vector<size_t> occs;
	if (!ptn_sz || ptn_sz > txt_sz) return occs;
	auto pi = prefFuncOf(ptn_beg, ptn_end);
	size_t i = 0, j = 0;
	auto iter = txt_beg;
	for (; i < txt_sz; ++i, ++iter) {
		while (j && *iter != *std::next(ptn_beg, j)) j = pi[j - 1];
		if (*iter == *std::next(ptn_beg, j)) ++j;
		if (j == ptn_sz) {
			occs.push_back(i + 1 - ptn_sz);
			j = pi[j - 1];
		}
	}
	return occs;
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
```

== Suffix Array

Let $Sigma$ be an alphabet,
and let $arrow(s) in Sigma^n space (n in NN)$ be a string.
An array $arrow(italic("sa")) in NN^n$ is called the *suffix array* of
the string $arrow(s)$ if and only if it satisfies that
for all $i in NN inter [0, n)$,
the suffix $s[italic("sa")_i, n)$ is the $i$-th lexicographically smallest suffix of $arrow(s)$.
While computing the suffix array,
we also maintain a *rank array* $arrow(italic("ra"))$,
such that for any $i in NN inter [0, n)$,
$italic("ra")_i$ is the lexicographical rank of the suffix $s[i, n)$ among all suffixes of $arrow(s)$.
Thus, we have
$
  italic("sa")_(italic("ra")_i) = i = italic("ra")_(italic("sa")_i).
$
Let $italic("lcp")(i, j)$ denote the length of the longest common prefix (LCP)
of the suffixes of $arrow(s)$ starting at $i$ and $j$, respectively.
The *height array* $arrow(h) in NN^n$ is defined as
$
  arrow(h) = (italic("lcp")(italic("sa")_(i-1), italic("sa")_i))_(i = 0)^(n - 1)
$
That is, $h_i$ is the LCP length of the suffix with rank $i$
and the suffix ranked just before it (rank $i - 1$).
Specifically, we define $h_0=0$.
Consequently,
$
  h_(italic("ra")_i)=italic("lcp")(italic("sa")_(italic("ra")_i), italic("sa")_(italic("ra")_i-1))=italic("lcp")(i, italic("sa")_(italic("ra")_i - 1))
$
In other words, $h_(italic("ra")_i)$ is the LCP length of the suffix
starting at $i$ and the suffix that is lexicographically ranked just before it.

#lemma[LCP Lemma][
  $h_(italic("ra")_i) >= h_(italic("ra")_(i - 1)) - 1$.
]<lem:lcp-lemma>

```cpp
/**
 * @return {sa, ra, height}
 */
template <typename Iter>
inline std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
suffArrOf(Iter begin, Iter end) {
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
			 i + len < n && j + len < n &&
			 *std::next(begin, i + len) == *std::next(begin, j + len);
			 ++len);
		height[ra[i]] = len;
	}

	return {sa, ra, height};
}
```

== Hash

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
	for (size_t i = 0; i != s.size(); ++i) h[i + 1] = (h[i] * b + s[i]) % mod;
	return h;
}
```

The hash value of a subarray $s_([b, e))$ can be efficiently computed as
$
  h(s_([b, e))) = (h(s_([0, e))) - h(s_([0, b))) * B^(e - b)) mod M,
$
#h(-indent) where $B$ is the base and $M$ is the modulus.

== Manacher

```
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
```

= Graph Theory

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

	Graph(size_t n = 0) : m_adjs_of(n) {}
	Graph(size_t n, std::vector<Edge> edges)
		: m_edges(std::move(edges)), m_adjs_of(n) { m_insertEdgesToAdj(); }

	void assign(size_t n) {
		m_edges.clear();
		m_adjs_of.resize(n);
		for (auto &lst : m_adjs_of) lst.clear();
	}
	void assign(size_t n, std::vector<Edge> edges) {
		m_edges = std::move(edges);
		m_adjs_of.resize(n);
		for (auto &lst : m_adjs_of) lst.clear();
		m_insertEdgesToAdj();
	}

	template <typename W = Weight>
	typename std::enable_if<std::is_void<W>::value, void>::type
	insertEdge(size_t u, size_t v) {
		m_edges.push_back(Edge{u, v});
		m_insertToAdj(m_edges.size() - 1);
	}
	template <typename W = Weight,
			  typename = typename std::enable_if<
				  std::is_same<W, Weight>::value>::type>
	typename std::enable_if<!std::is_void<W>::value, void>::type
	insertEdge(size_t u, size_t v, const W &w) {
		m_edges.push_back(Edge{u, v, w});
		m_insertToAdj(m_edges.size() - 1);
	}

	void reserveVerts(size_t n) { m_adjs_of.reserve(n); }
	void reserveEdges(size_t m) { m_edges.reserve(m); }

	void clear() {
		m_edges.clear();
		m_adjs_of.clear();
	}

	std::pair<size_t, size_t> size() const {
		return std::make_pair(m_adjs_of.size(), m_edges.size());
	}

	const std::vector<Edge> &edges() const { return m_edges; }
	const std::vector<std::vector<size_t>> &adjs_of_verts() const {
		return m_adjs_of;
	}

	/* Algorithms */
	/**
	 * @return {src, edges} where edges is the sequence of edge indices in the
	 * Eulerian trail starting from src. If no such Eulerian trail exists,
	 * return {size_t(-1), {}}.
	 */
	std::pair<size_t, std::vector<size_t>> hierholzer(size_t src = -1) const;

	std::vector<std::vector<size_t>> tarjanSccs() const;
	/**
	 * @return {{cut_verts, vbccs}, {bridges, ebccs}} of each connected component
	 *
	 * @note An isolated vertex is treated as a vbcc.
	 */
	inline std::vector<
		std::pair<std::pair<std::vector<size_t>,
							std::vector<std::vector<size_t>>>,
				  std::pair<std::vector<size_t>,
							std::vector<std::vector<size_t>>>>>
	tarjanCutAndBccs() const;

	std::vector<size_t> toposort() const;

	std::vector<std::vector<size_t>> kruskal() const;
	std::vector<size_t> prim(size_t rt = 0) const;

	std::vector<Weight> dijkstra(size_t src) const;
	std::vector<Weight> bellmanFord(size_t src) const;
	std::vector<Weight> spfa(size_t src) const;

	std::tuple<Weight, std::vector<bool>, std::vector<Weight>>
	dinic(size_t src, size_t dst,
		  Weight lim = std::numeric_limits<Weight>::max()) const;

protected:
	std::vector<Edge> m_edges;
	std::vector<std::vector<size_t>> m_adjs_of;

	void m_insertToAdj(size_t idx) {
		size_t sz_requirement = std::max(m_edges[idx].u, m_edges[idx].v) + 1;
		if (sz_requirement > m_adjs_of.size()) m_adjs_of.resize(sz_requirement);
		m_adjs_of[m_edges[idx].u].emplace_back(idx);
		if (m_edges[idx].u != m_edges[idx].v) {
			m_adjs_of[m_edges[idx].v].emplace_back(idx);
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

== Eulerian Path

#definition[Eulerian Trail][
  A #strong[trail] in a graph is called an #emph[Eulerian #underline[trail/path]] if it visits every edge exactly once.
]<def:eulerian-trail>

#definition[Eulerian Circuit][
  A #strong[circuit] in a graph is called an #emph[Eulerian circuit] if it is an Eulerian trail.
]<def:eulerian-circuit>

#definition[Eulerian Graph][
  A graph is called an #emph[Eulerian graph] if it contains an Eulerian circuit.
]<def:eulerian-graph>

#definition[Semi-Eulerian Graph][
  A graph is called a #emph[semi-Eulerian graph] if it contains an Eulerian trail but not an Eulerian circuit.
]<def:semi_eulerian-graph>

```cpp
template <typename Weight, bool is_directed>
std::pair<size_t, std::vector<size_t>>
Graph<Weight, is_directed>::hierholzer(size_t src) const {
	std::array<size_t, 2> ends{size_t(-1), size_t(-1)};
	if (is_directed) {
		std::vector<size_t> in_degs(m_adjs_of.size()), out_degs(m_adjs_of.size());
		for (const auto &edge : m_edges) {
			++out_degs[edge.u];
			++in_degs[edge.v];
		}

		for (size_t i = 0; i < m_adjs_of.size(); ++i) {
			if (in_degs[i] == out_degs[i]) continue;
			if (in_degs[i] + 1 == out_degs[i]) {
				if (~ends[0]) {
					return std::make_pair(size_t(-1), std::vector<size_t>());
				}
				ends[0] = i;
			} else if (in_degs[i] == out_degs[i] + 1) {
				if (~ends[1]) {
					return std::make_pair(size_t(-1), std::vector<size_t>());
				}
				ends[1] = i;
			} else {
				return std::make_pair(size_t(-1), std::vector<size_t>());
			}
		}
		if (~src && ~ends[0] && src != ends[0]) {
			return std::make_pair(size_t(-1), std::vector<size_t>());
		}
	} else {
		std::vector<size_t> degs(m_adjs_of.size());
		for (const auto &edge : m_edges) {
			++degs[edge.u];
			++degs[edge.v];
		}

		for (size_t i = 0; i < m_adjs_of.size(); ++i) {
			if (!(degs[i] & 1)) continue;
			if (~ends[0]) {
				if (~ends[1]) {
					return std::make_pair(size_t(-1), std::vector<size_t>());
				}
				ends[1] = i;
			} else {
				ends[0] = i;
			}
		}
		if (~src && ~ends[0] && src != ends[0] && src != ends[1]) {
			return std::make_pair(size_t(-1), std::vector<size_t>());
		}
	}
	if (static_cast<bool>(~ends[0]) != static_cast<bool>(~ends[1])) {
		return std::make_pair(size_t(-1), std::vector<size_t>());
	}
	if (src == size_t(-1)) {
		src = (~ends[0] ? ends[0] : 0);
	}
	assert(~src && src < m_adjs_of.size());

	std::vector<size_t> idxs(m_adjs_of.size());
	std::vector<bool> used(m_edges.size());
	std::vector<size_t> res;
	res.reserve(m_edges.size());
	std::function<void(size_t)> dfs = [&](size_t cur) -> void {
		for (auto &i = idxs[cur]; i < m_adjs_of[cur].size(); ++i) {
			auto e = m_adjs_of[cur][i];
			if (used[e]) continue;
			auto &edge = m_edges[e];
			if (is_directed && edge.u != cur) continue;
			auto nxt = ((edge.u == cur) ? edge.v : edge.u);
			used[e] = true;
			dfs(nxt);
			res.push_back(e);
		}
	};
	dfs(src);
	if (res.size() != m_edges.size()) { // not connected
		return std::make_pair(size_t(-1), std::vector<size_t>());
	}
	std::reverse(res.begin(), res.end());
	return std::make_pair(src, res);
}
```

== Connectivity

=== Tarjan for Strongly Connected Components

#definition[Strongly Connected][
  Two vertices $u$ and $v$ in a #strong[directed] graph is called #emph[strongly connected] if there exists a directed path from $u$ to $v$ and a directed path from $v$ to $u$.

  A directed graph $G$ is called #emph[strongly connected] if every pair of vertices in $G$ is strongly connected.

  A maximal strongly connected subgraph of a directed graph $G$ is called a #emph[strongly connected component (SCC)] of $G$.
]<def:strongly-connected>

```cpp
template <typename Weight, bool is_directed>
std::vector<std::vector<size_t>>
Graph<Weight, is_directed>::tarjanSccs() const {
	static_assert(is_directed, "Tarjan's algorithm for strongly connected components "
							   "is only applicable to directed graphs.");
	std::vector<size_t> stk;
	stk.reserve(m_adjs_of.size());
	std::vector<bool> in_stk(m_adjs_of.size());
	std::vector<size_t> dfn(m_adjs_of.size(), size_t(-1)), low(m_adjs_of.size());
	std::vector<std::vector<size_t>> sccs;
	size_t tm = 0;
	std::function<void(size_t)> dfs = [&](size_t cur) -> void {
		stk.emplace_back(cur);
		in_stk[cur] = true;
		low[cur] = (dfn[cur] = (tm++));
		for (auto adj : m_adjs_of[cur]) {
			auto &edge = m_edges[adj];
			if (edge.u != cur) { // edges pointing to cur
				continue;
			}
			auto nxt = edge.v;
			if (dfn[nxt] == size_t(-1)) {
				dfs(nxt);
				if (low[nxt] < low[cur]) low[cur] = low[nxt];
			} else if (in_stk[nxt]) {
				if (dfn[nxt] < low[cur]) low[cur] = dfn[nxt];
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
	for (size_t i = 0; i != m_adjs_of.size(); ++i) {
		if (dfn[i] == size_t(-1)) dfs(i);
	}
	return sccs;
}
```

The time and space complexity are both $Theta(|V| + |E|)$.

Tarjan's algorithm guarantees that the SCCs are numbered in the #strong[reverse topological order] of the DAG formed by contracting each SCC into a single vertex.

=== Tarjan for Cuts and Biconnected Components

#definition[Vertex Cut][
  Let $G = (V, E)$ be a #strong[strongly connected] graph. $V_"c" subset.eq V$ is called a #emph[#underline[vertex cut/separating set]] of $G$ if $G without V_"c"$ is #strong[not] strongly connected.

  Particularly, a vertex cut of size $1$ is called a #emph[cut vertex].
]<def:vertex-cut>

#definition[Vertex Connectivity][
  Let $G = (V, E)$ be a #strong[connected] graph, and $k in NN$. $G$ is called #emph[$k$-vertex-connected] if $|V| >= k + 1$ and there is no vertex cut of size $k - 1$. The maximum $k$ such that $G$ is $k$-vertex-connected is called the #emph[vertex connectivity] of $G$, denoted as $kappa(G)$.
]<vertex-connectivity>

#definition[Edge Cut][
  Let $G = (V, E)$ be a #strong[strongly connected] graph. $E_"c" subset.eq E$ is called a #emph[edge cut] of $G$ if $G without E_"c"$ is #strong[not] strongly connected.

  Particularly, an edge cut of size $1$ is called a #emph[bridge].
]<def:edge-cut>

Similarly, we can define #emph[edge connectivity] of a graph $G$, denoted as $lambda(G)$.

#definition[Biconnected][
  A connected graph is called #emph[biconnected] if it has no cut vertex.

  A maximal biconnected subgraph of a connected graph $G$ is called a #emph[biconnected component (BCC)] of $G$.
]<def:biconnected>

#definition[2-Edge-Connected][
  A connected graph is called #emph[2-edge-connected] if it has no bridge.

  A maximal 2-edge-connected subgraph of a connected graph $G$ is called a #emph[2-edge-connected component (2ECC)] of $G$.
]<def:2_edge_connected>

#theorem[Whitney Theorem][
  For any graph $G$,
  $
    kappa(G) <= lambda(G) <= delta(G),
  $
  where $delta(G) = min_(v in V(G)) deg(v)$.
]<thm:whitney-connectivity>

```cpp
template <typename Weight, bool is_directed>
inline std::vector<
	std::pair<std::pair<std::vector<size_t>,
						std::vector<std::vector<size_t>>>,
			  std::pair<std::vector<size_t>,
						std::vector<std::vector<size_t>>>>>
Graph<Weight, is_directed>::tarjanCutAndBccs() const {
	static_assert(!is_directed,
				  "Tarjan's algorithm for cut vertices, bridges "
				  "and biconnected components is only applicable to "
				  "undirected graphs.");
	std::vector<
		std::pair<std::pair<std::vector<size_t>,
							std::vector<std::vector<size_t>>>,
				  std::pair<std::vector<size_t>,
							std::vector<std::vector<size_t>>>>>
		ccs;
	std::vector<size_t> vbcc_stk, ebcc_stk;
	std::vector<size_t> dfn(m_adjs_of.size(), size_t(-1)), low(m_adjs_of.size());
	size_t tm = 0;
	std::function<void(size_t, size_t)> dfs = [&](size_t pre, size_t cur) {
		auto &cc = ccs.back();
		auto &cut_verts = cc.first.first;
		auto &vbccs = cc.first.second;
		auto &bridges = cc.second.first;
		auto &ebccs = cc.second.second;
		low[cur] = (dfn[cur] = (tm++));
		vbcc_stk.emplace_back(cur);
		ebcc_stk.emplace_back(cur);
		size_t children = 0;
		bool not_in_cut = true, has_edge_to_pre = false;
		for (auto i : m_adjs_of[cur]) {
			size_t nxt = ((m_edges[i].u == cur) ? m_edges[i].v : m_edges[i].u);
			if (dfn[nxt] == size_t(-1)) {
				++children;
				dfs(cur, nxt);
				if (low[nxt] < low[cur]) low[cur] = low[nxt];
				if (~pre ? (low[nxt] >= dfn[cur]) : (children > 1)) {
					if (not_in_cut) {
						not_in_cut = false;
						cut_verts.emplace_back(cur);
					}
					vbccs.emplace_back(1, cur);
					size_t vert;
					do {
						vbccs.back().push_back(vert = vbcc_stk.back());
						vbcc_stk.pop_back();
					} while (vert != nxt);
				}
				if (low[nxt] > dfn[cur]) {
					bridges.emplace_back(i);
					ebccs.emplace_back();
					size_t vert;
					do {
						ebccs.back().emplace_back(vert = ebcc_stk.back());
						ebcc_stk.pop_back();
					} while (vert != nxt);
				}
			} else if (nxt != pre || has_edge_to_pre) {
				if (dfn[nxt] < low[cur]) low[cur] = dfn[nxt];
			} else {
				has_edge_to_pre = true;
			}
		}
	};
	for (size_t i = 0; i != m_adjs_of.size(); ++i) {
		if (dfn[i] == size_t(-1)) {
			ccs.emplace_back();
			dfs(-1, i);
			if (vbcc_stk.size()) {
				auto &vbccs = ccs.back().first.second;
				vbccs.emplace_back(std::move(vbcc_stk));
				vbcc_stk.clear();
			}
			if (ebcc_stk.size()) {
				auto &ebccs = ccs.back().second.second;
				ebccs.emplace_back(std::move(ebcc_stk));
				ebcc_stk.clear();
			}
		}
	}
	return ccs;
}
```

The time and space complexity are both $Theta(|V| + |E|)$.

==== Round-Square Tree

#figure(
  caption: [Round-Square Tree],
)[
  #grid(
    columns: 3,
    image("figures/round_square-tree-1.svg"),
    image("figures/round_square-tree-2.svg"),
    image("figures/round_square-tree-3.svg"),
  )
]<fig:round_square-tree>

#definition[Round-Square Tree][
  Given an #strong[undirected connected] graph $G$, the #emph[round-suqare tree] $T$ of $G$ is constructed as follows:
  + Treat each vertex in $G$ as a #emph[round] vertex in $T$. Denote the set of all round vertices as $R = V(G)$.
  + Create a #emph[square] vertex in $T$ for each #strong[biconnected component]. Denote the set of all square vertices as $S$.
  + Build the tree $T$ over the vertex set $R union S$. For each $v in V(G)$ and each biconnected component $italic("vbcc") subset.eq V(G)$ s.t. $v in italic("vbcc")$ (note that one vertex may be included by multiple biconnected components), add an edge in $T$ between the round vertex corresponding to $v$ and the square vertex corresponding to $italic("vbcc")$.
]<def:round_square-tree>


== Topological Sort

```cpp
template <typename Weight, bool is_directed>
inline std::vector<size_t> Graph<Weight, is_directed>::toposort() const {
	static_assert(is_directed,
				  "Topological sorting is only applicable to directed graphs.");
	std::vector<size_t> indeg(m_adjs_of.size());
	for (auto &edge : m_edges) ++indeg[edge.v];
	std::vector<size_t> res, zero_indeg_verts;
	for (size_t i = 0; i != m_adjs_of.size(); ++i) {
		if (!indeg[i]) zero_indeg_verts.emplace_back(i);
	}
	while (zero_indeg_verts.size()) {
		size_t cur = zero_indeg_verts.back();
		zero_indeg_verts.pop_back();
		res.emplace_back(cur);
		for (auto adj : m_adjs_of[cur]) {
			auto nxt = m_edges[adj].v;
			if (!(--indeg[nxt])) {
				zero_indeg_verts.emplace_back(nxt);
			}
		}
	}
	return ((res.size() == m_adjs_of.size()) ? res : std::vector<size_t>());
}
```

== Minimum Spanning Tree and Forest

=== Kruskal

```cpp
template <typename Weight, bool is_directed>
inline std::vector<std::vector<size_t>>
Graph<Weight, is_directed>::kruskal() const {
	static_assert(!is_directed,
				  "Kruskal's algorithm is only applicable to undirected graphs.");
	if (m_adjs_of.size() < 2) return {};

	std::vector<size_t> sorted(m_edges.size());
	std::iota(sorted.begin(), sorted.end(), 0);
	std::sort(sorted.begin(), sorted.end(), [&](size_t lhs, size_t rhs) {
		return (m_edges[lhs].w < m_edges[rhs].w);
	});

	std::vector<size_t> dsu(m_adjs_of.size());
	std::iota(dsu.begin(), dsu.end(), 0);
	auto find = [&](size_t x) -> size_t {
		while (dsu[dsu[x]] != dsu[x]) dsu[x] = dsu[dsu[x]];
		return dsu[x];
	};
	auto merge = [&](size_t dst, size_t src) { dsu[find(src)] = find(dst); };

	std::vector<size_t> msf_edges;
	msf_edges.reserve(m_adjs_of.size() - 1);
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
```

=== Prim

```cpp
template <typename Weight, bool is_directed>
inline std::vector<size_t> Graph<Weight, is_directed>::prim(size_t rt) const {
	static_assert(!is_directed,
				  "Prim algorithm is only applicable to undirected graphs.");
	if (m_adjs_of.size() < 2) return {};

	std::vector<size_t> mst_edges;
	mst_edges.reserve(m_adjs_of.size() - 1);
	auto cmp = [&](size_t lhs, size_t rhs) {
		return (m_edges[lhs].w > m_edges[rhs].w);
	};
	std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> pq(cmp);
	std::vector<bool> vis(m_adjs_of.size());

	auto visit = [&](size_t cur) -> void {
		vis[cur] = true;
		for (auto adj : m_adjs_of[cur]) {
			const auto &edge = m_edges[adj];
			if (!vis[(edge.u == cur ? edge.v : edge.u)]) pq.push(adj);
		}
	};

	visit(rt);
	while (pq.size() && mst_edges.size() + 1 < m_adjs_of.size()) {
		size_t i = pq.top();
		pq.pop();
		size_t u = m_edges[i].u, v = m_edges[i].v;
		if (vis[u] && vis[v]) continue;
		mst_edges.emplace_back(i);
		visit(vis[u] ? v : u);
	}

	return mst_edges;
}
```

== Shortest Path

=== Floyd

```cpp
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
					weights[j][k] = std::min(weights[j][k],
											 weights[j][i] + weights[i][k]);
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
		auto n = lhs.size();
		Mat res(n, std::vector<Weight>(n, std::numeric_limits<Weight>::max()));
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
	Mat res(n, std::vector<Weight>(n, std::numeric_limits<Weight>::max()));
	for (size_t i = 0; i != n; ++i) res[i][i] = 0;
	for (; n; n >>= 1) {
		if (n & 1) res = matMul(res, weights);
		weights = matMul(weights, weights);
	}
	return res;
}
```

=== Dijkstra

```cpp
template <typename Weight, bool is_directed>
inline std::vector<Weight>
Graph<Weight, is_directed>::dijkstra(size_t src) const {
	using Adj = std::pair<Weight, size_t>; // (dist, vertex)
	std::priority_queue<Adj, std::vector<Adj>, std::greater<Adj>> pq;
	std::vector<Weight> dist(m_adjs_of.size(),
							 std::numeric_limits<Weight>::max());
	std::vector<bool> vis(m_adjs_of.size());

	pq.emplace(0, src);
	dist[src] = 0;
	while (pq.size()) {
		auto cur = pq.top().second;
		pq.pop();
		if (vis[cur]) continue;
		vis[cur] = true;
		for (auto adj : m_adjs_of[cur]) {
			const auto &edge = m_edges[adj];
			if (is_directed && cur != edge.u) continue;
			auto nxt = (is_directed
							? edge.v
							: ((edge.u == cur) ? edge.v : edge.u));
			auto w = edge.w;
			if (dist[nxt] > dist[cur] + w) {
				pq.emplace(dist[nxt] = dist[cur] + w, nxt);
			}
		}
	}

	return dist;
}
```

=== Bellman-Ford

```cpp
template <typename Weight, bool is_directed>
inline std::vector<Weight>
Graph<Weight, is_directed>::bellmanFord(size_t src) const {
	std::vector<Weight> dist(m_adjs_of.size(),
							 std::numeric_limits<Weight>::max());
	auto relax = [&](size_t u, size_t v, Weight w) -> bool {
		if (dist[u] != std::numeric_limits<Weight>::max() &&
			dist[v] > dist[u] + w) {
			dist[v] = dist[u] + w;
			return false;
		}
		return true;
	};

	dist[src] = 0;
	for (size_t cnt = 0; cnt != m_adjs_of.size(); ++cnt) {
		bool flag = true;
		for (auto &edge : m_edges) {
			flag &= relax(edge.u, edge.v, edge.w);
			if (!is_directed) flag &= relax(edge.v, edge.u, edge.w);
		}
		if (flag) return dist;
	}
	return {}; // negative cycle detected
}
```

=== SPFA

```cpp
template <typename Weight, bool is_directed>
inline std::vector<Weight>
Graph<Weight, is_directed>::spfa(size_t src) const {
	std::vector<Weight> dist(m_adjs_of.size(), std::numeric_limits<Weight>::max());
	std::vector<bool> inq(m_adjs_of.size());
	std::vector<size_t> cnt(m_adjs_of.size());
	std::queue<size_t> q;

	dist[src] = 0;
	q.emplace(src);
	while (q.size()) {
		auto cur = q.front();
		inq[cur] = false;
		q.pop();
		for (auto adj : m_adjs_of[cur]) {
			const auto &edge = m_edges[adj];
			if (is_directed && cur != edge.u) continue;
			auto nxt = (is_directed
							? edge.v
							: ((edge.u == cur) ? edge.v : edge.u));
			auto w = edge.w;
			if (dist[nxt] > dist[cur] + w) {
				dist[nxt] = dist[cur] + w;
				/**
				 * the shortest path between 2 vertices consists of at most
				 * (n - 1) edgess
				 */
				if ((cnt[nxt] = cnt[cur] + 1) >= m_adjs_of.size()) return {};
				if (!inq[nxt]) {
					inq[nxt] = true;
					q.emplace(nxt);
				}
			}
		}
	}
	return dist;
}
```

=== Shortest Hamiltonian Path and Cycle

#definition[Hamiltonian Path][
  A #strong[path] in a graph is called a #emph[Hamiltonian path] if it visits every vertex exactly once.
]<def:hamiltonian-path>

#definition[Hamiltonian Cycle][
  A #strong[cycle] in a graph is called a #emph[Hamiltonian cycle] if it is a Hamiltonian path.
]<def:hamiltonian-cycle>

```cpp
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
	std::vector<std::vector<Weight>> dp(uint64_t(1) << n,
										std::vector<Weight>(n, inf));
	std::vector<std::vector<size_t>> pre(uint64_t(1) << n,
										 std::vector<size_t>(n, size_t(-1)));
	dp[1][0] = 0;
	for (uint64_t set = 1; set < (uint64_t(1) << n); set += 2) {
		for (size_t cur = (set != 1); cur < n; ++cur) {
			if (!(set & (uint64_t(1) << cur))) continue;
			if (dp[set][cur] == inf) continue;
			for (size_t nxt = 1; nxt < n; ++nxt) {
				if ((set & (uint64_t(1) << nxt)) || adj_mat[cur][nxt] == inf) {
					continue;
				}
				uint64_t nxt_set = set | (uint64_t(1) << nxt);
				if (dp[set][cur] + adj_mat[cur][nxt] < dp[nxt_set][nxt]) {
					pre[nxt_set][nxt] = cur;
					dp[nxt_set][nxt] = dp[set][cur] + adj_mat[cur][nxt];
				}
			}
		}
	}
	uint64_t full_set = (uint64_t(1) << n) - 1;
	auto best = inf;
	size_t last = -1;
	for (size_t i = 1; i < n; ++i) {
		if (dp[full_set][i] == inf || adj_mat[i][0] == inf) continue;
		auto cost = dp[full_set][i] + adj_mat[i][0];
		if (cost < best) {
			best = cost;
			last = i;
		}
	}
	if (last == size_t(-1)) return {};
	std::vector<size_t> res;
	res.reserve(n);
	while (full_set) {
		res.emplace_back(last);
		size_t pre_last = pre[full_set][last];
		full_set ^= (uint64_t(1) << last);
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
	std::vector<std::vector<Weight>> dp(uint64_t(1) << n,
										std::vector<Weight>(n, inf));
	std::vector<std::vector<size_t>> pre(uint64_t(1) << n,
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
```

== Network Flow

=== Max Flow and Min Cut

```cpp
/**
 * @return {limited max flow, min cut, flows} from src to dst
 * - min_cut[i] == true iff i is unreachable from src in the residual network.
 * 	Particularly, the returned flow equals the unrestricted maximum flow iff
 * 	min_cut[dst] == true. In that case, min_cut describes a minimum cut, with
 * 	true denoting the sink side.
 *
 * @note Time complexity:
 * - General networks: O(|V|^2 * |E|).
 * - Unit-capacity networks (e.g., bipartite matching): O(|E| * min(|V|^{2/3},
 * 	sqrt(|E|))).
 * - Unit-capacity networks where every vertex except src and dst has either
 * 	in-degree 1 or out-degree 1: O(|E| * sqrt(|V|)).
 * In practice, Dinic's algorithm is extremely efficient and typically runs
 * much faster than its theoretical worst-case upper bound on real-world graphs.
 */
template <typename Weight, bool is_directed>
std::tuple<Weight, std::vector<bool>, std::vector<Weight>>
Graph<Weight, is_directed>::dinic(size_t src, size_t dst, Weight lim) const {
	assert(src != dst);
	assert(lim >= Weight());

	std::vector<Weight> caps(m_edges.size() << 1);
	std::vector<size_t> dep(m_adjs_of.size()), iter(m_adjs_of.size());
	std::deque<size_t> q;

	for (size_t e = 0; e < m_edges.size(); ++e) caps[e << 1] = m_edges[e].w;
	if (!is_directed) {
		for (size_t e = 0; e < m_edges.size(); ++e) caps[(e << 1) | 1] = m_edges[e].w;
	}

	auto bfs = [&]() -> void {
		std::fill(dep.begin(), dep.end(), size_t(-1));
		dep[src] = 0;
		q.clear();
		q.push_back(src);

		while (q.size()) {
			auto cur = q.front();
			q.pop_front();
			for (size_t e : m_adjs_of[cur]) {
				const auto &edge = m_edges[e];
				size_t nxt;
				if (edge.v != cur) {
					nxt = edge.v, e <<= 1;
				} else {
					nxt = edge.u, e = ((e << 1) | 1);
				}
				if (!caps[e] || ~dep[nxt]) continue;
				dep[nxt] = dep[cur] + 1;
				if (nxt == dst) return;
				q.push_back(nxt);
			}
		}
	};
	std::function<Weight(size_t, const Weight &)> dfs =
		[&](size_t cur, const Weight &up) -> Weight {
		if (cur == dst) return up;

		Weight flow = 0;
		for (auto &i = iter[cur]; i != m_adjs_of[cur].size(); ++i) {
			size_t nxt, e = m_adjs_of[cur][i];
			const auto &edge = m_edges[e];
			if (edge.v != cur) {
				nxt = edge.v, e = (e << 1);
			} else {
				nxt = edge.u, e = ((e << 1) | 1);
			}
			if (!caps[e] || dep[cur] >= dep[nxt]) continue;
			auto down = dfs(nxt, std::min(up - flow, caps[e]));
			if (!down) continue;
			flow += down;
			caps[e] -= down, caps[e ^ 1] += down;
			if (flow == up) return flow;
		}
		dep[cur] = m_adjs_of.size();
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

	std::vector<bool> min_cut(m_adjs_of.size(), true);
	q.clear();
	q.push_back(src);
	min_cut[src] = false;
	while (q.size()) {
		auto cur = q.front();
		q.pop_front();
		for (size_t e : m_adjs_of[cur]) {
			size_t nxt;
			if (m_edges[e].v != cur) {
				nxt = m_edges[e].v, e = (e << 1);
			} else {
				nxt = m_edges[e].u, e = ((e << 1) | 1);
			}
			if (caps[e] && min_cut[nxt]) {
				q.push_back(nxt);
				min_cut[nxt] = false;
			}
		}
	}

	std::vector<Weight> flows(m_edges.size());
	for (size_t e = 0; e < m_edges.size(); ++e) {
		flows[e] = m_edges[e].w - caps[e << 1];
	}

	return std::make_tuple(max_flow, std::move(min_cut), std::move(flows));
}
```

=== Min Cost Max Flow

```cpp
namespace atcoder {

namespace internal {

template <class E> struct csr {
    std::vector<int> start;
    std::vector<E> elist;
    explicit csr(int n, const std::vector<std::pair<int, E>>& edges)
        : start(n + 1), elist(edges.size()) {
        for (auto e : edges) {
            start[e.first + 1]++;
        }
        for (int i = 1; i <= n; i++) {
            start[i] += start[i - 1];
        }
        auto counter = start;
        for (auto e : edges) {
            elist[counter[e.first]++] = e.second;
        }
    }
};

}  // namespace internal

template <class Cap, class Cost> struct mcf_graph {
  public:
    mcf_graph() {}
    explicit mcf_graph(int n) : _n(n) {}

    int add_edge(int from, int to, Cap cap, Cost cost) {
        assert(0 <= from && from < _n);
        assert(0 <= to && to < _n);
        assert(0 <= cap);
        assert(0 <= cost);
        int m = int(_edges.size());
        _edges.push_back({from, to, cap, 0, cost});
        return m;
    }

    struct edge {
        int from, to;
        Cap cap, flow;
        Cost cost;
    };

    edge get_edge(int i) {
        int m = int(_edges.size());
        assert(0 <= i && i < m);
        return _edges[i];
    }
    std::vector<edge> edges() { return _edges; }

    std::pair<Cap, Cost> flow(int s, int t) {
        return flow(s, t, std::numeric_limits<Cap>::max());
    }
    std::pair<Cap, Cost> flow(int s, int t, Cap flow_limit) {
        return slope(s, t, flow_limit).back();
    }
    std::vector<std::pair<Cap, Cost>> slope(int s, int t) {
        return slope(s, t, std::numeric_limits<Cap>::max());
    }
    std::vector<std::pair<Cap, Cost>> slope(int s, int t, Cap flow_limit) {
        assert(0 <= s && s < _n);
        assert(0 <= t && t < _n);
        assert(s != t);

        int m = int(_edges.size());
        std::vector<int> edge_idx(m);

        auto g = [&]() {
            std::vector<int> degree(_n), redge_idx(m);
            std::vector<std::pair<int, _edge>> elist;
            elist.reserve(2 * m);
            for (int i = 0; i < m; i++) {
                auto e = _edges[i];
                edge_idx[i] = degree[e.from]++;
                redge_idx[i] = degree[e.to]++;
                elist.push_back({e.from, {e.to, -1, e.cap - e.flow, e.cost}});
                elist.push_back({e.to, {e.from, -1, e.flow, -e.cost}});
            }
            auto _g = internal::csr<_edge>(_n, elist);
            for (int i = 0; i < m; i++) {
                auto e = _edges[i];
                edge_idx[i] += _g.start[e.from];
                redge_idx[i] += _g.start[e.to];
                _g.elist[edge_idx[i]].rev = redge_idx[i];
                _g.elist[redge_idx[i]].rev = edge_idx[i];
            }
            return _g;
        }();

        auto result = slope(g, s, t, flow_limit);

        for (int i = 0; i < m; i++) {
            auto e = g.elist[edge_idx[i]];
            _edges[i].flow = _edges[i].cap - e.cap;
        }

        return result;
    }

  private:
    int _n;
    std::vector<edge> _edges;

    // inside edge
    struct _edge {
        int to, rev;
        Cap cap;
        Cost cost;
    };

    std::vector<std::pair<Cap, Cost>> slope(internal::csr<_edge>& g,
                                            int s,
                                            int t,
                                            Cap flow_limit) {
        // variants (C = maxcost):
        // -(n-1)C <= dual[s] <= dual[i] <= dual[t] = 0
        // reduced cost (= e.cost + dual[e.from] - dual[e.to]) >= 0 for all edge

        // dual_dist[i] = (dual[i], dist[i])
        std::vector<std::pair<Cost, Cost>> dual_dist(_n);
        std::vector<int> prev_e(_n);
        std::vector<bool> vis(_n);
        struct Q {
            Cost key;
            int to;
            bool operator<(Q r) const { return key > r.key; }
        };
        std::vector<int> que_min;
        std::vector<Q> que;
        auto dual_ref = [&]() {
            for (int i = 0; i < _n; i++) {
                dual_dist[i].second = std::numeric_limits<Cost>::max();
            }
            std::fill(vis.begin(), vis.end(), false);
            que_min.clear();
            que.clear();

            // que[0..heap_r) was heapified
            size_t heap_r = 0;

            dual_dist[s].second = 0;
            que_min.push_back(s);
            while (!que_min.empty() || !que.empty()) {
                int v;
                if (!que_min.empty()) {
                    v = que_min.back();
                    que_min.pop_back();
                } else {
                    while (heap_r < que.size()) {
                        heap_r++;
                        std::push_heap(que.begin(), que.begin() + heap_r);
                    }
                    v = que.front().to;
                    std::pop_heap(que.begin(), que.end());
                    que.pop_back();
                    heap_r--;
                }
                if (vis[v]) continue;
                vis[v] = true;
                if (v == t) break;
                // dist[v] = shortest(s, v) + dual[s] - dual[v]
                // dist[v] >= 0 (all reduced cost are positive)
                // dist[v] <= (n-1)C
                Cost dual_v = dual_dist[v].first, dist_v = dual_dist[v].second;
                for (int i = g.start[v]; i < g.start[v + 1]; i++) {
                    auto e = g.elist[i];
                    if (!e.cap) continue;
                    // |-dual[e.to] + dual[v]| <= (n-1)C
                    // cost <= C - -(n-1)C + 0 = nC
                    Cost cost = e.cost - dual_dist[e.to].first + dual_v;
                    if (dual_dist[e.to].second - dist_v > cost) {
                        Cost dist_to = dist_v + cost;
                        dual_dist[e.to].second = dist_to;
                        prev_e[e.to] = e.rev;
                        if (dist_to == dist_v) {
                            que_min.push_back(e.to);
                        } else {
                            que.push_back(Q{dist_to, e.to});
                        }
                    }
                }
            }
            if (!vis[t]) {
                return false;
            }

            for (int v = 0; v < _n; v++) {
                if (!vis[v]) continue;
                // dual[v] = dual[v] - dist[t] + dist[v]
                //         = dual[v] - (shortest(s, t) + dual[s] - dual[t]) +
                //         (shortest(s, v) + dual[s] - dual[v]) = - shortest(s,
                //         t) + dual[t] + shortest(s, v) = shortest(s, v) -
                //         shortest(s, t) >= 0 - (n-1)C
                dual_dist[v].first -= dual_dist[t].second - dual_dist[v].second;
            }
            return true;
        };
        Cap flow = 0;
        Cost cost = 0, prev_cost_per_flow = -1;
        std::vector<std::pair<Cap, Cost>> result = {{Cap(0), Cost(0)}};
        while (flow < flow_limit) {
            if (!dual_ref()) break;
            Cap c = flow_limit - flow;
            for (int v = t; v != s; v = g.elist[prev_e[v]].to) {
                c = std::min(c, g.elist[g.elist[prev_e[v]].rev].cap);
            }
            for (int v = t; v != s; v = g.elist[prev_e[v]].to) {
                auto& e = g.elist[prev_e[v]];
                e.cap += c;
                g.elist[e.rev].cap -= c;
            }
            Cost d = -dual_dist[s].first;
            flow += c;
            cost += c * d;
            if (prev_cost_per_flow == d) {
                result.pop_back();
            }
            result.push_back({flow, cost});
            prev_cost_per_flow = d;
        }
        return result;
    }
};

}  // namespace atcoder
```
