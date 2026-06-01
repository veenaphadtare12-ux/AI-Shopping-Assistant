import { useState, useRef } from 'react'

function App() {
  const [query, setQuery] = useState('')
  const [maxPrice, setMaxPrice] = useState(5000)
  const [minRating, setMinRating] = useState(0)
  
  // Platform selection state, without Meesho
  const [platforms, setPlatforms] = useState({
    Amazon: true,
    Flipkart: true,
    Myntra: true
  })
  
  const [loading, setLoading] = useState(false)
  const [products, setProducts] = useState([])
  const [bestPick, setBestPick] = useState(null)
  const [aiSummary, setAiSummary] = useState('')
  const [error, setError] = useState('')
  const [sortMethod, setSortMethod] = useState('relevance')
  
  const resultsRef = useRef(null)

  const handlePlatformChange = (p) => {
    setPlatforms(prev => ({ ...prev, [p]: !prev[p] }))
  }

  const handleSearch = async (e, directQuery = null) => {
    if (e) e.preventDefault()
    const searchQuery = directQuery || query
    if (!searchQuery.trim()) return

    setLoading(true)
    setError('')
    setProducts([])
    setBestPick(null)
    setAiSummary('')

    const activePlatforms = Object.entries(platforms)
      .filter(([_, active]) => active)
      .map(([name]) => name)

    const body = {
      query: searchQuery,
      max_price: maxPrice,
      min_rating: minRating,
      platforms: activePlatforms,
      limit: 10
    }

    try {
      const res = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || res.statusText)
      }

      const data = await res.json()
      setAiSummary(data.ai_summary || '')
      setBestPick(data.best_pick || null)
      
      const normalizedProducts = (data.products || []).map(p => ({
        id: p.product_id || Math.random().toString(),
        name: p.product_name || p.name || 'Product',
        brand: p.platform || '',
        category: p.category || '',
        rating: p.rating || 0,
        reviews: p.rating_count || p.review_count || 0,
        price: p.discounted_price || p.price || 0,
        originalPrice: p.mrp || null,
        ai_explanation: p.ai_explanation || ''
      }))
      
      setProducts(normalizedProducts)
      
      if (resultsRef.current) {
        resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    } catch (err) {
      setError('Search failed: ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleChipClick = (text) => {
    setQuery(text)
    handleSearch(null, text)
  }

  const getSortedProducts = () => {
    let sorted = [...products]
    if (sortMethod === 'price-asc') sorted.sort((a, b) => a.price - b.price)
    if (sortMethod === 'price-desc') sorted.sort((a, b) => b.price - a.price)
    if (sortMethod === 'rating') sorted.sort((a, b) => b.rating - a.rating)
    return sorted
  }

  return (
    <>
      <nav>
        <div className="logo">Shop<span>Mind</span></div>
        <div className="nav-links">
          <a href="#">Deals</a>
          <a href="#">Compare</a>
          <a href="#">Trending</a>
          <a href="#">Watchlist</a>
        </div>
        <div className="nav-badge">AI Powered</div>
      </nav>

      <section className="hero">
        <div className="hero-tag">
          <svg viewBox="0 0 24 24">
            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
          </svg>
          Smart AI Recommendations
        </div>

        <h1>Find the <em>perfect product</em><br/>in seconds</h1>
        <p className="hero-sub">
          Describe what you're looking for — our AI finds the best matches<br/>
          with real-time price comparisons across top stores.
        </p>

        <div className="search-wrap">
          <form className="search-box" onSubmit={handleSearch}>
            <div className="search-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8"/>
                <path d="m21 21-4.35-4.35"/>
              </svg>
            </div>
            <input 
              id="searchInput" 
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Try 'wireless headphones' or 'running shoes'…" 
            />
            <button type="submit" className="search-btn" disabled={loading}>
              {loading ? 'Searching…' : 'Search'}
            </button>
          </form>

          <div className="search-controls">
            <div style={{ flex: 1, minWidth: '200px' }}>
              <label>
                Max price: <span>₹{maxPrice}</span>
              </label>
              <input 
                type="range" 
                min="100" max="20000" step="50" 
                value={maxPrice} 
                onChange={(e) => setMaxPrice(Number(e.target.value))} 
              />
            </div>

            <div style={{ minWidth: '160px' }}>
              <label>Min rating</label>
              <div style={{ display: 'flex', gap: '6px' }}>
                {[0, 3, 4, 5].map(r => (
                  <button 
                    key={r} type="button" 
                    className={`chip ${minRating === r ? 'active' : ''}`}
                    onClick={() => setMinRating(r)}
                  >
                    {r === 0 ? 'Any' : r === 5 ? '5★' : `${r}★+`}
                  </button>
                ))}
              </div>
            </div>

            <div style={{ minWidth: '200px' }}>
              <label>Platforms</label>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {Object.keys(platforms).map(p => (
                  <label key={p} className="platform-checkbox">
                    <input 
                      type="checkbox" 
                      checked={platforms[p]} 
                      onChange={() => handlePlatformChange(p)} 
                    /> 
                    {p}
                  </label>
                ))}
              </div>
            </div>
          </div>

          <div className="chips">
            {['Wireless Earbuds', 'Mechanical Keyboard', 'Gaming Mouse', 'Smartwatch', 'Laptop Stand'].map(chip => (
              <div key={chip} className="chip" onClick={() => handleChipClick(chip)}>
                {chip}
              </div>
            ))}
          </div>
        </div>

        <div className="stats">
          <div className="stat">
            <div className="stat-num">2.4M+</div>
            <div className="stat-label">Products indexed</div>
          </div>
          <div className="stat">
            <div className="stat-num">340+</div>
            <div className="stat-label">Stores tracked</div>
          </div>
          <div className="stat">
            <div className="stat-num">Real-time</div>
            <div className="stat-label">Price updates</div>
          </div>
          <div className="stat">
            <div className="stat-num">AI</div>
            <div className="stat-label">Powered matching</div>
          </div>
        </div>
      </section>

      <section className="results-section" ref={resultsRef} style={{ display: loading || products.length > 0 || error ? 'block' : 'none' }}>
        
        {error && (
          <div style={{ marginBottom: '18px', background: '#fee2e2', color: '#991b1b', padding: '12px', borderRadius: '8px' }}>
            {error}
          </div>
        )}

        {aiSummary && (
          <div className="ai-summary-card">
            {aiSummary}
          </div>
        )}

        {bestPick && (
          <div className="card best-pick-card" onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            e.currentTarget.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
            e.currentTarget.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div className="best-pick-badge">Best Pick</div>
                <div className="best-pick-title">{bestPick.product_name || bestPick.name}</div>
                <div className="best-pick-subtitle">{bestPick.platform} · Rating {bestPick.rating}★</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <div className="best-pick-price">₹{bestPick.discounted_price || bestPick.price}</div>
                <a href="#" className="best-pick-buy">Buy</a>
              </div>
            </div>
            {bestPick.ai_explanation && (
              <div className="best-pick-explanation">{bestPick.ai_explanation}</div>
            )}
          </div>
        )}

        {(loading || products.length > 0) && (
          <div className="results-header">
            <div className="results-title">
              Recommendations <span>({products.length} found)</span>
            </div>
            <select className="sort-select" value={sortMethod} onChange={(e) => setSortMethod(e.target.value)}>
              <option value="relevance">Sort: Relevance</option>
              <option value="price-asc">Price: Low to High</option>
              <option value="price-desc">Price: High to Low</option>
              <option value="rating">Top Rated</option>
            </select>
          </div>
        )}

        <div className="grid">
          {loading && Array(6).fill(0).map((_, i) => (
            <div key={i} className="skeleton">
              <div className="skel-line" style={{ width: '40%', height: '10px' }}></div>
              <div className="skel-line" style={{ width: '80%', height: '14px', marginTop: '16px' }}></div>
              <div className="skel-line" style={{ width: '60%', height: '10px' }}></div>
              <div className="skel-line" style={{ width: '30%', height: '20px', marginTop: '24px' }}></div>
            </div>
          ))}

          {!loading && products.length === 0 && query && !error && (
            <div className="empty">
              <span className="empty-icon">🔍</span>
              <h3>No products found</h3>
              <p>Try different keywords or browse our trending picks</p>
            </div>
          )}

          {!loading && getSortedProducts().map((p, i) => (
            <div 
              key={p.id} 
              className="card" 
              style={{ animationDelay: `${i * 0.07}s` }}
              onMouseMove={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                e.currentTarget.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
                e.currentTarget.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
              }}
            >
              <div className="card-top">
                <div className="card-cat">{p.category}</div>
                <div className="card-rating">
                  <svg viewBox="0 0 24 24">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                  </svg>
                  {p.rating}
                  <span style={{ opacity: .5 }}>({(p.reviews / 1000).toFixed(1)}k)</span>
                </div>
              </div>

              <div className="card-name">{p.name}</div>
              <div className="card-brand">{p.brand}</div>

              <div className="card-bottom">
                <div>
                  <span className="card-price">₹{p.price}</span>
                  {p.originalPrice && <span className="card-price-original">₹{p.originalPrice}</span>}
                </div>
                <button className="card-btn" title="Add to watchlist">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06
                             a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78
                             1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                  </svg>
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      <footer>
        <div className="footer-logo">Shop<span>Mind</span></div>
        <div>© 2026 ShopMind.</div>
        <div>Privacy · Terms · API</div>
      </footer>
    </>
  )
}

export default App
