print('hello from lua (script file)')

zenc_hello('test2')

local tbl = {
    ["something"] = "good",
    [2] = 3
}

for k,v in pairs(tbl) do
    print("Key: "..k.." | Val: "..v)
end

local str = "4"
print(tonumber(str))

print(os.date("today is %A, in %B"))
