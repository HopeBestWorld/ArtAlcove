DROP TABLE IF EXISTS episodes;

CREATE TABLE episodes(
    id int,
    title varchar(64),
    descr varchar(1024)
);

CREATE TABLE stretched_canvas(
    id int AUTO_INCREMENT PRIMARY KEY,
    product varchar(250),
    descr varchar(1024),
    siteurl varchar(200) UNIQUE,
    price dec(6, 2),
    rating dec(2, 1),
    imgurl varchar(200),
    price_range varchar(50)
);

CREATE TABLE stretched_canvas_reviews(
    id int AUTO_INCREMENT PRIMARY KEY,
    review_title varchar(200),
    review_desc varchar(1024),
    product varchar(250),
    FOREIGN KEY(product) REFERENCES stretched_canvas(product)
);

CREATE TABLE alcohol_markers(
    id int AUTO_INCREMENT PRIMARY KEY,
    product varchar(250),
    descr varchar(1024),
    siteurl varchar(200) UNIQUE,
    price dec(6, 2),
    rating dec(2, 1),
    imgurl varchar(200),
    price_range varchar(50)
);

CREATE TABLE alcohol_markers_reviews(
    id int AUTO_INCREMENT PRIMARY KEY,
    review_title varchar(200),
    review_desc varchar(1024),
    product varchar(250),
    FOREIGN KEY(product) REFERENCES alcohol_markers(product)
);