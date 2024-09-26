# SQL Para Análise de Dados e Data Science - Capítulo 09


-- Lista os pedidos
SELECT * FROM cap09.dsa_pedidos;


-- Média do valor dos pedidos
SELECT AVG(valor_pedido) AS media
FROM cap09.dsa_pedidos;


-- Média do valor dos pedidos
SELECT ROUND(AVG(valor_pedido), 2) AS media
FROM cap09.dsa_pedidos;


-- Média do valor dos pedidos por cidade
SELECT cidade_cliente, ROUND(AVG(valor_pedido), 2) AS media
FROM cap09.dsa_pedidos P, cap09.dsa_clientes C
WHERE P.id_cliente = C.id_cli
GROUP BY cidade_cliente;


-- Média do valor dos pedidos por cidade ordenado pela media
SELECT cidade_cliente, ROUND(AVG(valor_pedido), 2) AS media
FROM cap09.dsa_pedidos P, cap09.dsa_clientes C
WHERE P.id_cliente = C.id_cli
GROUP BY cidade_cliente
ORDER BY media DESC;


-- Média do valor dos pedidos por cidade com INNER JOIN
SELECT cidade_cliente, ROUND(AVG(valor_pedido),2) AS media
FROM cap09.dsa_pedidos P 
INNER JOIN cap09.dsa_clientes C ON P.id_cliente = C.id_cli
GROUP BY cidade_cliente
ORDER BY media DESC;


-- Insere um novo registro na tabela de Clientes
INSERT INTO cap09.dsa_clientes (id_cli, nome_cliente, tipo_cliente, cidade_cliente, estado_cliente) 
VALUES (1011, 'Agatha Christie', 'Ouro', 'Belo Horizonte', 'MG');


-- Média do valor dos pedidos por cidade com INNER JOIN
SELECT cidade_cliente, ROUND(AVG(valor_pedido),2) AS media
FROM cap09.dsa_pedidos P 
INNER JOIN cap09.dsa_clientes C ON P.id_cliente = C.id_cli
GROUP BY cidade_cliente
ORDER BY media DESC;


-- Média do valor dos pedidos por cidade (mostrar cidades sem pedidos)
SELECT cidade_cliente,ROUND(AVG(valor_pedido),2) AS media
FROM cap09.dsa_pedidos P 
RIGHT JOIN cap09.dsa_clientes C ON P.id_cliente = C.id_cli
GROUP BY cidade_cliente
ORDER BY media DESC;


-- Média do valor dos pedidos por cidade (mostrar cidades sem pedidos) - erro
SELECT cidade_cliente, COALESCE(ROUND(AVG(valor_pedido),2), 'Não Houve Pedido') AS media
FROM cap09.dsa_pedidos P 
RIGHT JOIN cap09.dsa_clientes C ON P.id_cliente = C.id_cli
GROUP BY cidade_cliente
ORDER BY media DESC;


-- Média do valor dos pedidos por cidade (mostrar cidades sem pedidos)
SELECT 
    cidade_cliente, 
    CASE 
        WHEN AVG(valor_pedido) IS NULL THEN 'Não Houve Pedido' 
        ELSE CAST(ROUND(AVG(valor_pedido), 2) AS VARCHAR)
    END AS media
FROM 
    cap09.dsa_clientes C
    LEFT JOIN cap09.dsa_pedidos P ON C.id_cli = P.id_cliente
GROUP BY 
    cidade_cliente
ORDER BY media DESC;


-- Média do valor dos pedidos por cidade (mostrar cidades sem pedidos)
SELECT 
    cidade_cliente, 
    CASE 
        WHEN AVG(valor_pedido) IS NULL THEN 'Não Houve Pedido' 
        ELSE CAST(ROUND(AVG(valor_pedido), 2) AS VARCHAR)
    END AS media
FROM 
    cap09.dsa_clientes C
    LEFT JOIN cap09.dsa_pedidos P ON C.id_cli = P.id_cliente
GROUP BY 
    cidade_cliente
ORDER BY 
    CASE 
        WHEN AVG(valor_pedido) IS NULL THEN 1 
        ELSE 0 
    END, 
    media DESC;


-- Média do valor dos pedidos por cidade (mostrar cidades sem pedidos e com valor 0)
SELECT  
    cidade_cliente,
    CASE 
        WHEN ROUND(AVG(valor_pedido),2) IS NULL THEN 0
        ELSE ROUND(AVG(valor_pedido),2)
    end AS media
FROM cap09.dsa_pedidos P 
RIGHT JOIN cap09.dsa_clientes C ON P.id_cliente = C.id_cli
GROUP BY cidade_cliente
ORDER BY media DESC;

