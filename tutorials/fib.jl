using ResumableFunctions
using Infiltrator

@resumable function fib()
    n = BigInt(0)
    m = BigInt(1)
    while true
        m_new = n + m
        n = m
        m = m_new
        @show n, m, m_new
        @yield m_new
    end
end

function printer()
    gen = fib()
    for _ in 1:20
        x = gen()
    end
end

printer()


